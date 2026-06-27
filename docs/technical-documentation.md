# Technical Documentation

AIgnition 3.0 — Probabilistic Revenue & ROAS Forecasting

## Forecasting methodology

We forecast e-commerce **revenue** and **ROAS** as probabilistic ranges (P10/P50/P90)
over 30/60/90-day aggregate windows, at four levels: total (blended), channel,
campaign-type, and campaign.

The model is a **hierarchical Bayesian media-response model** fit per
`channel × campaign-type × campaign` series. Each series is decomposed as:

```
revenue_t = baseline_t + saturation(spend_t)
baseline_t = level + seasonality(day-of-week)
saturation(spend) = alpha * spend^slope / (kappa^slope + spend^slope)
```

- **Baseline** captures the organic level and weekly seasonality (the brief
  explicitly asks for seasonality in campaign-type forecasting).
- **Saturation (Hill curve)** maps incremental spend to incremental revenue with
  diminishing returns. This is what makes budget simulation principled: doubling
  spend yields *saturating* returns, not a linear extrapolation.
- **Observation noise** is log-normal (heavy-tailed, positive), robust to spikes.

**Probabilistic outputs** come from posterior-predictive simulation: for each of
thousands of posterior parameter draws we simulate a daily revenue path over the
horizon, sum to the aggregate window, and read empirical quantiles. Blended ROAS
is computed per draw as `Σ revenue / Σ spend` (a ratio of sums, never an average
of ratios), then reduced to quantiles.

**The offline-compile trick.** Bayesian inference (PyMC) runs once offline in
`train.py`. We then extract the posterior draws and fitted components as plain
NumPy arrays and pickle a thin `BayesianForecaster` whose `predict()` is pure
NumPy/SciPy. At scoring time the pipeline loads that pickle and simulates with no
PyMC, no MCMC, and no network — deterministic under a fixed seed. This keeps the
scored `requirements.txt` minimal (numpy/scipy/pandas/pyarrow), which is the
single biggest defense against the cross-version unpickling failures the
submission guide warns about.

## Model selection

We considered three approaches:

1. **Quantile gradient boosting** — fast and robust but tree models cannot
   extrapolate beyond historical spend, which breaks budget simulation.
2. **Hierarchical Bayesian response model (chosen)** — naturally probabilistic,
   naturally budget-conditional via the saturation curve, and explainable
   parameter-by-parameter, which grounds the AI narrative.
3. **Hybrid decomposition + conformal** — a strong middle ground; its uncertainty
   ideas (coverage checks, conformal recalibration) are retained as a validation
   and calibration layer on top of the Bayesian core.

Partial pooling (global → channel → campaign-type priors) lets sparse or noisy
campaigns borrow strength from their parents — directly addressing the brief's
"fragmented, inconsistent data" problem.

## Data preprocessing logic

The inputs are the **ad-platform performance exports** — Google Ads, Microsoft
(Bing) Ads, and Meta Ads — carrying spend and conversion value per campaign.
(Per the organizer Q&A, GA4 and Shopify files are **not** used.) All monetary
values are **USD**.

1. **Ingestion** (`forecast_core/ingest.py`) reads every CSV under `data/`
   dynamically (by pattern, never hardcoded filenames) and normalizes each to a
   canonical long frame: `date, channel, campaign_type, campaign, spend, revenue,
   conversions`. Column names are resolved through a configurable `COLUMN_MAP`,
   so adapting to the real dataset is a one-dict edit. It is **robust**: extra
   columns are ignored, missing optional columns are filled, and files without a
   date+monetary signal are skipped rather than crashing.
2. **Channel resolution** maps a platform/source column (or, when absent, the
   file name) to `google / microsoft / meta / other` via `CHANNEL_ALIASES` /
   `FILENAME_CHANNEL_HINTS`. Platform-specific quirks handled: Google spend is
   given in **micros** (`metrics_cost_micros` ÷ 1e6); campaign-type labels are
   normalized so platforms align (`PERFORMANCE_MAX`, `PerformanceMax` →
   `performance_max`); the Meta export has **no revenue or campaign-type column**,
   so Meta's `conversion` (conversion value) is used as revenue and its
   campaign type is derived from the campaign name.
3. **Validation** (`forecast_core/validate.py`) checks campaign consistency — a
   `(channel, campaign)` pair must map to a single campaign type (campaign names
   may legitimately repeat across channels) — plus non-negative revenue/spend and
   required columns; the pipeline fails loudly on blocking issues.
4. **Feature frame** (`forecast_core/features.py`) aggregates to daily per
   `(channel, campaign_type, campaign)` series and adds day-of-week and
   week-index seasonality indices.

## Mathematical formulation

For each series *s* = (channel, campaign type, campaign) and day *t*:

```
revenue_{s,t} = baseline_{s,t} + Hill(spend_{s,t})         (mean)
baseline_{s,t} = level_s + seasonal_s[dow(t)]
Hill(x) = alpha_s · x^{slope_s} / (kappa_s^{slope_s} + x^{slope_s})
revenue_{s,t} ~ LogNormal( log(mean), sigma_s )            (likelihood)
```

Parameters `(level_s, seasonal_s, alpha_s, kappa_s, slope_s, sigma_s)` are given
hierarchical priors (global → channel → campaign-type), inferred by MCMC offline.
For a horizon *H* and a future daily spend plan, the posterior-predictive total is

```
R_s = Σ_{t=1..H} revenue_{s,t}        (per posterior draw d → R_s^{(d)})
```

simulated in pure NumPy over all draws. Aggregation to any level *L* sums the
member series per draw, and ROAS is a **ratio of sums**:

```
Revenue_L^{(d)} = Σ_{s∈L} R_s^{(d)}
ROAS_L^{(d)}    = Σ_{s∈L} R_s^{(d)}  /  Σ_{s∈L} Spend_s
```

Reported P10/P50/P90 are the 10th/50th/90th percentiles of the draw vector
`{·^{(d)}}`. Because `slope ≤ … ` and the Hill curve is concave for typical
fits, doubling spend yields **less-than-double** incremental revenue — the
diminishing-returns behavior agencies expect.

## Assumptions

- Inputs are the **ad-platform exports** (Google / Microsoft / Meta Ads) with
  spend and conversion value; GA4 and Shopify are not used. Values are **USD**.
- **Meta revenue assumption:** the Meta export has no revenue column, so its
  `conversion` field is treated as conversion value (revenue). Per-row
  `conversion ÷ spend` lands at a sane ROAS (~2), supporting this reading rather
  than a conversion count. To revise, change the `revenue` candidates in
  `COLUMN_MAP` — a one-line edit.
- ROAS is undefined for paused (zero-spend) campaigns, so those campaign-level
  ROAS cells are left blank; revenue is always produced.
- A campaign's identity is the `(channel, campaign)` pair — campaign names can
  repeat across platforms, so the whole pipeline keys series by `channel::campaign`.
- "Total / blended" headline metrics sum the three paid channels; `campaign_type`
  metrics aggregate that type across channels.
- Spend basis: the `spend`/`cost` column drives ROAS; forecasts are conditioned on
  a future daily spend plan (recent run-rate by default, or user-supplied budgets
  in the app).
- Output columns follow `forecast_core/output_schema.py` — the single source of
  truth, aligned to the official format in one edit.
- Existing channel-level attribution is taken as the source of truth (no custom
  attribution or full MMM, per the brief). Test data may use different clients'
  campaigns; ingestion is built to absorb that.

## Limitations

- Adstock/carryover is not modeled by default (kept as a stretch; depends on data
  granularity).
- Forecasts are aggregate-period, not daily (by design and per the constraints).
- Forecast quality depends on history length; very short histories widen
  intervals and weaken seasonality estimates.
- The committed model is built with the no-PyMC fallback compiler for
  out-of-the-box runs; the full Bayesian fit (`--method bayesian`) should be run
  on the real dataset for the strongest results.

## AI integration strategy

The LLM layer (Anthropic Claude) is used **only in the demo app**, never on the
scored path. Claude receives **structured diagnostics** computed by the model —
decomposition contributions, per-channel saturation status (headroom vs
saturated), scenario deltas, and the headline revenue/ROAS quantiles — and is
instructed to interpret, not invent, numbers. It returns a causal narrative,
operational risk flags, and budget recommendations. A deterministic
`EchoProvider` backs the same interface so the app and its tests run offline
without an API key.
