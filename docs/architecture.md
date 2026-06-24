# Architecture Overview

## Two-track design

A single Bayesian model is trained once offline and compiled into a
dependency-light pickled predictor. A shared `forecast_core` package is consumed
by two tracks so they can never drift apart:

```
        input data/ (Google/Microsoft/Meta ads csv)   train.py (PyMC, offline, once)
                        |                                   |
                        |                                   v
                        |                            pickle/model.pkl
                        |                          (compiled posterior, numpy)
                        v                                   |
              +-------------------------------------------------------+
              |            forecast_core (shared engine)              |
              | ingest · validate · features · response curves ·      |
              | posterior simulation · aggregation · output schema    |
              +-------------------------------------------------------+
                        |                                   |
        Track 1 (offline, scored)               Track 2 (human-judged demo)
        run.sh -> generate_features             FastAPI backend -> Next.js app
              -> predict -> predictions.csv      -> /insights -> Claude
```

## Frontend stack

- **Next.js 14 (App Router) + React 18 + TypeScript + Tailwind CSS.**
- **Recharts** for P10–P90 fan charts.
- Four screens: upload + validation, forecast dashboard (30/60/90 toggle,
  channel drilldown, blended-ROAS range), budget simulator (per-channel sliders,
  base vs scenario), and an AI insights panel.
- A typed API client (`frontend/lib/api.ts`) wraps the backend.

## Backend stack

- **FastAPI + Uvicorn + Pydantic.**
- Endpoints: `POST /upload`, `GET /validate`, `POST /forecast`,
  `POST /simulate`, `POST /insights`.
- `ForecastService` wraps `forecast_core`, so the API and the scored `run.sh`
  share the exact same forecasting code.

## Forecasting pipeline

1. `generate_features.py` — ingest → validate → daily feature frame
   (`features.parquet`).
2. `predict.py` — load `pickle/model.pkl`, run posterior-predictive simulation
   for all horizons/levels/metrics, aggregate, write `predictions.csv`.
3. `run.sh` orchestrates both in one offline, deterministic, seeded invocation.

## LLM integration workflow

1. The app calls `ForecastService.diagnostics()` to compute a structured summary
   (no raw data).
2. `claude_insights.build_prompt()` serializes the diagnostics into a grounded
   prompt with strict instructions not to invent numbers.
3. The selected provider (`ClaudeProvider` when `ANTHROPIC_API_KEY` is set, else
   the offline `EchoProvider`) returns JSON with narrative, risks, and
   recommendations, surfaced in the insights panel.
