from __future__ import annotations
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ForecastRequest
from app.services.forecast_service import ForecastService
from app.llm.provider import EchoProvider, ClaudeProvider
from app.llm.claude_insights import generate_insights
from forecast_core import ingest, validate

app = FastAPI(title="AIgnition Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

MODEL_PATH = os.environ.get("MODEL_PATH", "pickle/model.pkl")
_svc = ForecastService(MODEL_PATH)
_UPLOAD_DIR = tempfile.mkdtemp(prefix="aignition_data_")


def _provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeProvider()
    return EchoProvider()


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    for f in files:
        dest = os.path.join(_UPLOAD_DIR, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
    return {"saved": [f.filename for f in files], "dir": _UPLOAD_DIR}


@app.get("/validate")
def validate_uploaded():
    rep = validate.validate_campaigns(ingest.load_data(_UPLOAD_DIR))
    return {"ok": rep.ok, "n_rows": rep.n_rows, "n_campaigns": rep.n_campaigns,
            "date_min": rep.date_min, "date_max": rep.date_max, "issues": rep.issues}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    return {"rows": _svc.forecast(req.horizon, req.budget_plan)}


@app.post("/simulate")
def simulate(req: ForecastRequest):
    return {"rows": _svc.forecast(req.horizon, req.budget_plan),
            "diagnostics": _svc.diagnostics(req.horizon, req.budget_plan)}


@app.post("/insights")
def insights(req: ForecastRequest):
    diag = _svc.diagnostics(req.horizon, req.budget_plan)
    return generate_insights(diag, _provider())
