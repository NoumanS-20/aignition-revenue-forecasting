# AIgnition 3.0 — Probabilistic Revenue Forecasting: Design Spec

- **Date:** 2026-06-23
- **Status:** Approved (design) — pending implementation plan
- **Hackathon:** NetElixir AIgnition 3.0
- **Key dates:** Start 2026-06-12 · Submission deadline 2026-07-19 22:00 IST · Finalists 2026-07-27 · Final presentation 2026-08-05
- **Submit to:** sunitha.k@netelixir.us (public GitHub repo URL + exact run command + team details)

## 1. Problem & context

Build an AI-assisted utility that produces **probabilistic forecasts** of e-commerce **revenue** and **blended ROAS** from historical marketing + sales data, with channel/campaign-type/campaign-level breakdowns, budget-scenario simulation, and AI-generated causal explanations.

Inputs (per the brief): GA4 session source/medium conversion data + Shopify conversion data. Paid channels in scope: Google Ads, Microsoft Ads, Meta Ads. Existing channel-level attribution is the source of truth — building a custom attribution engine or full MMM is out of scope.

The three pain points we explicitly target:
1. Forecasts that ignore ROAS constraints (we forecast ROAS as a first-class metric).
2. Fragmented, inconsistent, manually-tagged multi-channel data (we use hierarchical partial pooling + a validation layer).
3. Forecasts that don't explain *why* (we attach a model-grounded causal narrative).

## 2. The central constraint (why the architecture is shaped this way)

Two source documents impose conflicting requirements:

- **Project Brief** wants an ambitious product: probabilistic ranges, budget simulation, an LLM causal-inference layer, a frontend + backend, and a live demo.
- **Submission Guide** defines automated scoring: clone repo → `pip install -r requirements.txt` → drop in held-out test data → run `./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>` → read `predictions.csv`. **No internet at run time. No retraining. Pre-trained pickled model only. Must unpickle cleanly across library versions.** Any failure in this sequence = **zero** for that run.

**Implication:** the project splits into two tracks sharing one engine:

- **Track 1 — Scored pipeline (automated, offline, zero-tolerance).** Deterministic feature-gen + pickled model → `predictions.csv` in the exact announced format, no network, no LLM. This earns the automated score and gates everything.
- **Track 2 — Demo layer (human-judged).** Frontend + FastAPI backend + Claude causal narratives + budget simulator. This wins Practical Relevance, AI Integration, and Product Thinking at the demo.

Ambition is aimed at Track 2; Track 1 stays deliberately boring and bulletproof.

## 3. Goals / non-goals

**Goals**
- Probabilistic (P10/P50/P90) aggregate-period forecasts for 30/60/90-day windows.
- Metrics: total e-commerce revenue, blended ROAS, plus channel / campaign-type / campaign-level revenue contribution and ROAS ranges.
- Budget-conditional forecasting: accept future media budgets and re-forecast with saturating returns.
- Hierarchical Bayesian model, trained offline, compiled to a dependency-light picklable predictor.
- Claude-generated causal summaries, anomaly interpretation, and operational risk flags.
- A submission that passes the automated contract on a clean clone, first try.

**Non-goals (YAGNI)**
- No custom attribution engine; no full MMM. Use attribution as-is.
- No daily-granularity forecast deliverable — outputs are aggregate-period.
- No multi-tenant SaaS, auth, billing, or user accounts.
- No live model training or LLM calls inside the scored pipeline.

## 4. Evaluation-criteria mapping

| Criterion | How we win it |
|---|---|
| Technical soundness | Hierarchical Bayesian response model; calibrated intervals; rolling-origin backtest with coverage/pinball/WAPE. |
| Practical relevance | Budget simulator + ROAS-first outputs aligned to real agency planning workflows. |
| AI integration | Claude consumes structured model diagnostics → grounded causal narrative, anomaly interpretation, risk flags. |
| Product thinking | Clean four-screen app: ingest → forecast → simulate → insights. |
| Engineering quality | Shared `forecast_core`, two-track design, tests incl. a `run.sh` smoke test, pinned deps, full docs. |

## 5. Architecture overview

A single hierarchical Bayesian model is trained once offline and "compiled" into a lightweight pickled predictor. A shared `forecast_core` package is consumed by both tracks:

- **Inputs:** `data/` folder (GA4 + Shopify CSVs).
- **Offline training:** `train.py` (PyMC/NumPyro) → `pickle/model.pkl` (compiled posterior as NumPy arrays).
- **Shared engine:** `forecast_core` — ingest, validation, feature generation, response curves, posterior-predictive simulation, hierarchical aggregation, output schema.
- **Track 1:** `run.sh` → `generate_features.py` → `predict.py` → `predictions.csv` (offline, deterministic).
- **Track 2:** FastAPI backend (reusing `forecast_core`) → Next.js frontend; `/insights` calls Claude.

## 6. The offline-Bayesian "compile" trick

This is what lets us be genuinely Bayesian without violating the scoring contract.

1. `train.py` performs full Bayesian inference in PyMC/NumPyro and obtains a posterior (thousands of draws over all parameters: trend, seasonality coefficients, hierarchical channel/campaign effects, Hill saturation parameters, noise scale).
2. We **extract** the posterior draws and fitted components as plain NumPy arrays plus metadata (channel/campaign index maps, seasonality basis definitions, horizon config) and pickle a thin `BayesianForecaster` class.
3. `BayesianForecaster.predict()` performs **posterior-predictive simulation in pure NumPy/SciPy** — given a horizon and a future-spend plan, it simulates revenue paths per draw and reduces them to P10/P50/P90. No PyMC import, no MCMC, no network, deterministic under a fixed seed.

Consequences:
- **Scored `requirements.txt` is tiny** (`numpy`, `scipy`, `pandas`, plus minimal IO), minimizing the cross-version unpickling risk the guide flags as the #1 failure cause.
- PyMC/NumPyro and other heavy training deps live in a separate `requirements-train.txt` the judges never install or run.
- The pickle contains only arrays + plain Python objects + our own class — robust to load.

## 7. Forecasting methodology (Track 1 core)

**Model:** hierarchical Bayesian media-response model fit at channel × campaign-type (× campaign where data supports it).

- **Hierarchy / partial pooling:** global → channel → campaign-type priors. Sparse or noisy series borrow strength from their parent, stabilizing the fragmented-data case.
- **Structure per series:** `revenue_t = baseline_t + Σ_channel saturation(spend_{c,t}; θ_c)` where
  - `baseline_t = trend_t + seasonality_t`; trend = random-walk or penalized spline; seasonality = Fourier terms (weekly always; yearly/holiday if history length allows).
  - `saturation(spend; θ)` = Hill / diminishing-returns curve giving incremental revenue with saturation. (Adstock/carryover optional stretch — only if data granularity supports it.)
- **Likelihood:** heavy-tailed (Student-t on log-revenue or Gamma) for robustness to spikes/outliers.
- **Probabilistic outputs:** posterior-predictive simulation over the horizon → empirical P10/P50/P90 for revenue and ROAS, at every requested level and window.
- **ROAS computation:** per posterior draw, `blended_ROAS = Σ revenue / Σ spend` over the horizon, then take quantiles. Never average ratios. Channel/campaign ROAS computed at their own level analogously.
- **Hierarchical aggregation:** forecast at the finest level; sum revenue up to campaign-type, channel, and blended/total; recompute ROAS at each level. Optional forecast reconciliation as a stretch.
- **Budget conditioning:** future spend defaults to a recent run-rate; user/test-supplied budgets override per channel and flow through the saturation curves, so doubling spend yields saturating (not linear) incremental revenue with honest uncertainty.

**Validation (the evidence judges want):**
- Rolling-origin backtest. Report **interval coverage** (e.g. does the 80% band contain truth ~80% of the time), **pinball loss** for quantiles, and **WAPE/MAPE** for point accuracy.
- Optional **conformal recalibration** layer to correct interval coverage on a holdout if the raw Bayesian intervals are miscalibrated.

## 8. Output schema (designed for an unknown official format)

The exact `predictions.csv` columns were announced at launch and we do not yet have them. We absorb this risk:

- A single module `output_schema.py` is the **one source of truth** for column names, ordering, and row layout. Matching the official format becomes a one-file edit.
- **Default tidy layout** (placeholder until official format obtained): one row per `(level, entity, horizon_days, metric, p10, p50, p90)` where `level ∈ {total, channel, campaign_type, campaign}`, `metric ∈ {revenue, roas}`, `horizon_days ∈ {30, 60, 90}`.
- Writer is format-agnostic so a wide/pivoted official format is a thin adapter, not a rewrite.

## 9. Scored pipeline contract (Track 1 packaging)

- `run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>` at repo root, executable and `bash run.sh`-runnable, with the guide's defaults (`./data`, `./pickle/model.pkl`, `./output/predictions.csv`), `set -euo pipefail`, no prompts.
- One invocation runs feature-gen then predict end-to-end.
- **Schema-adaptive ingestion:** read `data/` by documented pattern / folder iteration, never hardcode row counts or unverified filenames; tolerate a different number of records.
- **Campaign-consistency validation:** check channel/campaign naming and structural consistency; emit a validation report; fail loudly on contract violations.
- Reproducibility: global seed set; relative paths only; no network calls; fresh output written each run.

## 10. Backend + Claude causal layer (Track 2)

- **FastAPI** reusing `forecast_core`. Endpoints: `/upload`, `/validate`, `/forecast`, `/simulate` (budget scenarios), `/insights` (Claude).
- **Claude (latest Opus/Sonnet via the Anthropic SDK)** consumes **structured diagnostics, not raw data**: decomposition contributions (trend vs seasonality vs each channel), per-channel saturation status (headroom vs saturated), detected anomalies (actuals outside the posterior band), scenario deltas, and backtest coverage. It returns a grounded causal narrative + operational risk flags + recommended budget reallocation.
- Provider abstracted behind one interface for safety, defaulting to Claude.
- LLM usage is demo-only; it is never on the scored path.

## 11. Frontend (Track 2)

**Next.js + React + Tailwind**, charting via a fan-chart-capable library. Four screens:
1. **Upload & validation** — drop CSVs, see the consistency report.
2. **Forecast dashboard** — P10–P90 fan charts for revenue & blended ROAS, 30/60/90 toggle, channel/campaign-type/campaign drilldown.
3. **Budget simulator** — per-channel sliders/inputs → live re-forecast → scenario A/B compare + recommended allocation.
4. **AI insights** — Claude narrative, anomaly cards, risk flags, "explain this forecast".

## 12. Repository structure

```
<repo>/
├── run.sh                      # Track 1 entry point (required)
├── requirements.txt            # scored deps only: numpy, scipy, pandas (pinned)
├── requirements-train.txt      # PyMC/NumPyro etc. (training only)
├── requirements-app.txt        # FastAPI, anthropic, etc. (demo only)
├── data/                       # small committed sample; overwritten at test
├── pickle/
│   └── model.pkl               # compiled posterior predictor (required)
├── src/
│   ├── generate_features.py    # CLI: data/ -> features.parquet
│   ├── predict.py              # CLI: features + model -> predictions.csv
│   └── forecast_core/          # shared engine
│       ├── ingest.py
│       ├── validate.py
│       ├── features.py
│       ├── response_curves.py
│       ├── bayesian_predict.py # BayesianForecaster (pure numpy)
│       ├── uncertainty.py      # calibration / conformal
│       ├── aggregate.py
│       └── output_schema.py
├── train.py                    # offline Bayesian fit -> pickle/model.pkl
├── app/                        # FastAPI backend (demo only)
│   └── llm/claude_insights.py
├── frontend/                   # Next.js app (demo only)
├── tests/                      # unit + run.sh smoke test
├── docs/                       # technical doc, architecture, methodology
└── README.md
```

## 13. Reproducibility & engineering quality

- Pin every dependency version across all three requirements files. State Python version in README.
- Seeds everywhere randomness affects predictions; deterministic NumPy RNG for posterior-predictive simulation.
- No absolute paths; no network at scored run time.
- **Tests:** unit tests for response curves, hierarchical aggregation, and output schema; a backtest harness; a smoke test that runs `run.sh` against committed sample data on a clean checkout and asserts the output contract.
- Optional Dockerfile only after confirming with organizers that Docker submissions are accepted.

## 14. Deliverables mapping

| Required deliverable | Produced by |
|---|---|
| Working prototype | `run.sh` scored pipeline + the demo app |
| Technical documentation | `docs/` — methodology, model selection, preprocessing, assumptions, limitations, AI strategy |
| Architecture overview | `docs/` — frontend/backend/forecasting pipeline/LLM workflow + the two-track diagram |
| Demo workflow | Walkthrough: ingestion → forecast → budget simulation → AI insights |

## 15. Priority phasing (must-win core vs stretch)

Given the ~4-week timeline, build in this order so a submittable artifact exists early:

- **P0 (must-win, gates the score):** schema-adaptive ingest + validation; feature generation; a working `BayesianForecaster.predict()` (even from a simple first fit); `run.sh` producing a valid `predictions.csv`; the `run.sh` smoke test passing on a clean clone.
- **P1 (rigor + headline features):** full hierarchical Bayesian fit; saturation curves + budget conditioning; backtest with coverage/pinball/WAPE; conformal recalibration.
- **P2 (demo wins):** FastAPI + Claude insights; Next.js dashboard, fan charts, budget simulator.
- **P3 (stretch):** adstock/carryover; forecast reconciliation; scenario optimizer / recommended allocation; Bayesian-vs-baseline comparison in docs.

## 16. Risks & open questions

1. **#1 risk — wrong output format.** The official `predictions.csv` schema and the dataset were announced at launch and are not yet in hand. **External action item: obtain the AIgnition dataset + exact output format ASAP.** Mitigation: `output_schema.py` single-source + schema-adaptive ingest.
2. **Pickle portability.** Mitigated by compiling the posterior to NumPy and keeping scored deps minimal and pinned; verify load on a clean environment.
3. **Bayesian fit time / convergence.** Mitigated by phasing — a simple fit unblocks P0; richer hierarchy added in P1. NumPyro (JAX) as a faster sampler option if PyMC is slow.
4. **Data granularity for seasonality/adstock.** Depends on history length; seasonality terms and adstock are gated on what the data supports.
5. **Docker acceptance.** Confirm with organizers before relying on it; default to pinned `requirements.txt`.

## 17. IP note

Per AIgnition 3.0 terms, all submitted IP becomes NetElixir's property and submissions must be original work. No third-party code/data used without rights.
