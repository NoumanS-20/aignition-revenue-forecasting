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

1. **Ingestion** (`forecast_core/ingest.py`) reads every CSV under `data/`
   dynamically (by pattern, never hardcoded filenames). GA4-style conversion
   files are detected by column presence and normalized to a canonical long
   frame: `date, channel, campaign_type, campaign, spend, revenue, conversions`.
   Column names are resolved through a configurable `COLUMN_MAP`, so adapting to
   the real dataset is a one-dict edit.
2. **Channel classification** maps source/medium to `google / microsoft / meta /
   other` via `CHANNEL_RULES`.
3. **Validation** (`forecast_core/validate.py`) checks campaign consistency
   (a campaign must map to one channel), non-negative revenue/spend, required
   columns, and emits a report; the pipeline fails loudly on blocking issues.
4. **Feature frame** (`forecast_core/features.py`) aggregates to daily per series
   and adds day-of-week and week-index seasonality indices.

## Assumptions

- "Total / blended" headline metrics sum the three paid channels (google,
  microsoft, meta); other channels are still forecast at the channel level.
- Spend basis: a `spend`/`cost` column is used when present; otherwise the recent
  per-channel run-rate (and, in the app, user-supplied budgets) is the spend
  basis for ROAS.
- Output columns follow `forecast_core/output_schema.py`. This is the single
  source of truth and will be aligned to the official launch format in one edit.
- Existing channel-level attribution is taken as the source of truth (no custom
  attribution or full MMM, per the brief).

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
