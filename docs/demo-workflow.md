# Demo Workflow

A four-step walkthrough mapping to the brief's required demo (data ingestion →
forecast generation → budget simulation → AI insights).

## Setup

```bash
# Backend (repo root)
pip install -r requirements-app.txt && pip install -e .
uvicorn app.main:app --port 8000
# optional: export ANTHROPIC_API_KEY=... to use real Claude instead of the offline stub

# Frontend (separate shell)
cd frontend && npm install && npm run dev
# open http://localhost:3000
```

## 1 · Data ingestion

Upload `data/sample/ga4.csv` and `data/sample/shopify.csv` (or the real dataset)
in the **Upload data** panel. The app calls `/upload` then `/validate` and shows
row count, campaign count, date range, and any consistency issues. Screenshot:
the green "Validation passed" report.

## 2 · Forecast generation

In **Forecast**, toggle 30/60/90 days. The fan chart shows each channel's revenue
P10–P90 band with the P50 line; the blended ROAS range is shown above it.
Screenshot: the fan chart at 90 days plus the blended-ROAS range.

## 3 · Budget simulation

In **Budget simulator**, move the per-channel sliders and click **Re-forecast**.
The app calls `/simulate`; the base→scenario revenue P50 updates, demonstrating
saturating returns from the response curves. Screenshot: a before/after where
increasing a saturated channel adds little while a channel with headroom adds
more.

## 4 · AI-generated business insights

Click **Explain this forecast** in **AI insights**. The app calls `/insights`,
which feeds structured diagnostics to Claude and renders the causal narrative,
risk flags, and budget recommendations. Screenshot: the narrative plus the risks
and recommendations lists.

## Scored pipeline (no UI)

To show the graded artifact:

```bash
bash run.sh ./data/sample ./pickle/model.pkl ./output/predictions.csv
head output/predictions.csv
```
