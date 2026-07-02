# AIgnition — Probabilistic Revenue & ROAS Forecasting

An AI-assisted forecasting utility for e-commerce marketing. It produces
**probabilistic** (P10/P50/P90) forecasts of revenue and blended ROAS over
30/60/90-day windows at total / channel / campaign-type / campaign level, supports
**budget-scenario simulation** via saturation response curves, and generates
**AI causal insights** with Anthropic Claude.

**Inputs:** ad-platform performance exports — Google Ads, Microsoft (Bing) Ads,
and Meta Ads — with spend and conversion value per campaign. All amounts in
**USD**. (GA4 and Shopify files are not used.)

- **Python:** 3.11+ (developed and tested on 3.12).
- **Scored pipeline is fully offline** — no network access required at run time.

## Scored pipeline (the graded artifact)

```bash
pip install -r requirements.txt
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` accepts three positional arguments with sensible defaults
(`./data`, `./pickle/model.pkl`, `./output/predictions.csv`). It generates
features from whatever is in the data folder, loads the pre-trained pickled model,
and writes predictions — in one command, with no internet and no retraining.

Output columns: `level, entity, horizon_days, metric, p10, p50, p90`
(single source of truth in `src/forecast_core/output_schema.py`).

## Retraining the model (optional, offline)

The committed `pickle/model.pkl` is the **full Bayesian (NUTS) fit** on the
provided dataset — it stores only relative (channel, campaign-type) shapes, so
it applies to whatever campaigns are present in `data/` at run time.

```bash
# Full Bayesian fit (used for the committed model):
pip install -r requirements-train.txt && pip install -e .
python train.py --data-dir data --out pickle/model.pkl --method bayesian --draws 500

# Fast method-of-moments fallback (no PyMC needed):
python train.py --data-dir data/sample --out pickle/model.pkl --method fallback

# One-shot helper (installs deps, fits, verifies run.sh):
bash scripts/train_bayesian.sh data 500
```

> The likelihood is collapsed onto per-weekday sufficient statistics (see
> `docs/technical-documentation.md`), so the full MCMC fit takes ~5 minutes even
> without a C compiler (set `PYTENSOR_FLAGS="cxx="` on Windows). A Colab
> notebook is also provided: [`notebooks/train_bayesian_colab.ipynb`](notebooks/train_bayesian_colab.ipynb).

## Demo app

```bash
pip install -r requirements-app.txt && pip install -e .
uvicorn app.main:app --port 8000          # backend
cd frontend && npm install && npm run dev  # frontend -> http://localhost:3000
```

Set `ANTHROPIC_API_KEY` to use real Claude for insights; without it the app uses
a deterministic offline stub.

## Tests

```bash
pip install -r requirements-app.txt && pip install -e .
pytest -q
```

## Project layout

- `src/forecast_core/` — shared engine (ingest, validate, features, response
  curves, Bayesian predictor, aggregation, uncertainty, output schema).
- `src/generate_features.py`, `src/predict.py`, `run.sh` — scored pipeline.
- `train.py` — offline trainer (Bayesian + fallback).
- `app/` — FastAPI backend + Claude insights.
- `frontend/` — Next.js demo.
- `docs/` — technical documentation, architecture overview, demo workflow.

## Documentation

- [Technical documentation](docs/technical-documentation.md)
- [Architecture overview](docs/architecture.md)
- [Demo workflow](docs/demo-workflow.md)
