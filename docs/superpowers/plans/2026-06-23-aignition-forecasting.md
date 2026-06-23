# AIgnition Probabilistic Revenue Forecasting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-trained hierarchical Bayesian forecasting utility that emits probabilistic revenue + ROAS predictions via a one-command pipeline, plus a Claude-powered demo app for budget simulation and causal insight.

**Architecture:** One Bayesian model is fit offline (PyMC) and "compiled" to a dependency-light pure-NumPy posterior predictor pickled into `pickle/model.pkl`. A shared `forecast_core` package powers both an offline, deterministic scored pipeline (`run.sh` → `predictions.csv`) and a FastAPI + Next.js demo (fan charts, budget sliders, Claude narratives). The scored path never imports PyMC, the Anthropic SDK, or touches the network.

**Tech Stack:** Python 3.11, NumPy/SciPy/pandas/pyarrow (scored path); PyMC/ArviZ (training only); FastAPI/Uvicorn/Anthropic SDK (backend); Next.js/React/Tailwind/Recharts (frontend).

## Global Constraints

These apply to **every** task. Verbatim from the spec.

- **Scored path is offline:** no network calls at run time; no PyMC, no `anthropic`, no LLM on the `run.sh` path.
- **No retraining at score time:** the model is pre-trained and committed; `run.sh` only generates features and predicts.
- **`run.sh` contract:** at repo root, named exactly `run.sh`, executable and runnable via `bash run.sh`; accepts `./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>` with defaults `./data`, `./pickle/model.pkl`, `./output/predictions.csv`; `set -euo pipefail`; one invocation runs feature-gen then predict; no interactive input.
- **`data/` is overwritten at test time** with a held-out set of the same schema: read it dynamically (by pattern / folder iteration), never hardcode row counts; tolerate a different number of records.
- **Pin every dependency version** in all requirements files. Python version stated in README (3.11).
- **Reproducibility:** global seed `SEED = 42`; deterministic NumPy RNG via `forecast_core.config.get_rng`; relative paths only; output written fresh each run.
- **Pickle portability:** `model.pkl` contains only NumPy arrays + plain Python objects + the `BayesianForecaster` class — no PyMC objects. Scored `requirements.txt` stays minimal.
- **Git hygiene:** `.gitignore` must NOT exclude `pickle/model.pkl` or committed sample data; no Git LFS (judges do a plain `git clone`).
- **Forecast levels:** `total`, `channel`, `campaign_type`, `campaign`. **Metrics:** `revenue`, `roas`. **Horizons:** 30, 60, 90 days. **Quantiles:** P10/P50/P90.
- **Documented assumptions** (carried because the dataset/output format were not in hand at plan time):
  - "Total / blended" headline metric = sum over the three paid channels (`google`, `microsoft`, `meta`); other channels present in GA4 are still forecast at channel level.
  - Spend basis: a `spend`/`cost` column is used if present; otherwise the channel run-rate (and, in the app, user-supplied budgets) is the spend basis for ROAS. Captured in `column_map`.
  - Output column layout follows `forecast_core.output_schema` (default tidy layout) until the official format is published — then it is a one-file change.

---

## File structure (created across the plan)

```
NetElixr/
├── run.sh                          # T13  scored entry point
├── requirements.txt                # T1   scored deps (pinned, minimal)
├── requirements-train.txt          # T1   PyMC/ArviZ (training only)
├── requirements-app.txt            # T1   FastAPI/anthropic (demo only)
├── .gitignore                      # T1   with pickle/ + sample data un-ignored
├── README.md                       # T25
├── data/sample/                    # T2   committed synthetic sample CSVs
├── pickle/model.pkl                # T14  compiled posterior (built by train.py)
├── train.py                        # T14  offline Bayesian fit -> pickle
├── src/
│   ├── generate_features.py        # T11  CLI: data/ -> features.parquet
│   ├── predict.py                  # T12  CLI: features+model -> predictions.csv
│   └── forecast_core/
│       ├── __init__.py             # T1
│       ├── config.py               # T2   constants, RNG, column_map
│       ├── ingest.py               # T3   schema-adaptive loader
│       ├── validate.py             # T4   campaign-consistency report
│       ├── features.py             # T5   feature frame + parquet IO
│       ├── response_curves.py      # T6   Hill saturation (numpy)
│       ├── bayesian_predict.py     # T7   BayesianForecaster (pure numpy)
│       ├── aggregate.py            # T8   hierarchical aggregation + ROAS
│       ├── uncertainty.py          # T9   coverage/pinball/WAPE + conformal
│       └── output_schema.py        # T10  predictions df + writer
├── app/
│   ├── main.py                     # T16  FastAPI app
│   ├── schemas.py                  # T15  pydantic models
│   ├── services/forecast_service.py# T15  wraps forecast_core
│   └── llm/
│       ├── provider.py             # T17  LLM provider abstraction
│       └── claude_insights.py      # T17  diagnostics -> narrative
├── frontend/                       # T19-23  Next.js app
├── docs/
│   ├── technical-documentation.md  # T24
│   ├── architecture.md             # T24
│   └── demo-workflow.md            # T25
└── tests/
    ├── conftest.py                 # T2
    ├── test_ingest.py              # T3
    ├── test_validate.py            # T4
    ├── test_features.py            # T5
    ├── test_response_curves.py     # T6
    ├── test_bayesian_predict.py    # T7
    ├── test_aggregate.py           # T8
    ├── test_uncertainty.py         # T9
    ├── test_output_schema.py       # T10
    ├── test_predict_cli.py         # T12
    ├── test_run_sh_smoke.py        # T13
    └── app/test_endpoints.py       # T16/T18
```

---

# PHASE 0 — Foundation

## Task 1: Repository scaffolding, dependency files, git hygiene

**Files:**
- Create: `requirements.txt`, `requirements-train.txt`, `requirements-app.txt`, `.gitignore`, `src/forecast_core/__init__.py`, `pyproject.toml`

**Interfaces:**
- Produces: an installable `forecast_core` package importable as `from forecast_core import ...` (via `pip install -e .`), three pinned requirements files.

- [ ] **Step 1: Create the package marker and pyproject so imports resolve**

`src/forecast_core/__init__.py`:
```python
"""Shared forecasting engine for the AIgnition submission."""
__version__ = "0.1.0"
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "forecast-core"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the pinned scored-path requirements**

`requirements.txt` (versions are a starting point — after `pip install`, lock to whatever resolved on the build machine and re-test):
```
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
pyarrow==16.1.0
python-dateutil==2.9.0.post0
```

`requirements-train.txt`:
```
-r requirements.txt
pymc==5.16.2
arviz==0.18.0
```

`requirements-app.txt`:
```
-r requirements.txt
fastapi==0.111.0
uvicorn==0.30.1
anthropic==0.34.2
pydantic==2.7.4
python-multipart==0.0.9
pytest==8.2.2
httpx==0.27.0
```

- [ ] **Step 3: Write `.gitignore` that protects the model and sample data**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
output/
features.parquet
node_modules/
frontend/.next/
.env
# Keep these tracked even though build artifacts are ignored above:
!pickle/
!pickle/model.pkl
!data/
!data/sample/
```

- [ ] **Step 4: Verify install + import works**

Run:
```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash
pip install -e . -r requirements.txt
python -c "import forecast_core; print(forecast_core.__version__)"
```
Expected: prints `0.1.0`, exit 0.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml requirements*.txt .gitignore src/forecast_core/__init__.py
git commit -m "chore: scaffold forecast_core package and pinned deps"
```

---

## Task 2: Config, deterministic RNG, column map, and a synthetic sample dataset

**Files:**
- Create: `src/forecast_core/config.py`, `tests/conftest.py`, `scripts/make_sample_data.py`, `data/sample/ga4.csv`, `data/sample/shopify.csv`
- Test: `tests/conftest.py` (fixtures)

**Interfaces:**
- Produces:
  - `config.SEED: int = 42`, `config.get_rng(seed=SEED) -> np.random.Generator`
  - `config.PAID_CHANNELS = ("google","microsoft","meta")`, `config.HORIZONS = (30,60,90)`, `config.LEVELS = ("total","channel","campaign_type","campaign")`, `config.METRICS = ("revenue","roas")`, `config.QUANTILES = (0.10,0.50,0.90)`
  - `config.COLUMN_MAP: dict` describing assumed source columns
  - `pytest` fixture `sample_data_dir` → path to a temp dir holding generated CSVs
  - `scripts/make_sample_data.py:generate(out_dir, n_days=180, seed=42)` writing `ga4.csv` + `shopify.csv`

- [ ] **Step 1: Write the failing test for config + RNG determinism**

`tests/test_config.py`:
```python
from forecast_core import config

def test_constants_present():
    assert config.SEED == 42
    assert config.HORIZONS == (30, 60, 90)
    assert config.LEVELS == ("total", "channel", "campaign_type", "campaign")
    assert config.QUANTILES == (0.10, 0.50, 0.90)

def test_rng_is_deterministic():
    a = config.get_rng().normal(size=5)
    b = config.get_rng().normal(size=5)
    assert (a == b).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError` (config not defined).

- [ ] **Step 3: Implement `config.py`**

`src/forecast_core/config.py`:
```python
from __future__ import annotations
import numpy as np

SEED: int = 42
PAID_CHANNELS: tuple[str, ...] = ("google", "microsoft", "meta")
HORIZONS: tuple[int, ...] = (30, 60, 90)
LEVELS: tuple[str, ...] = ("total", "channel", "campaign_type", "campaign")
METRICS: tuple[str, ...] = ("revenue", "roas")
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Assumed source-column names. Editing this dict adapts ingestion to the
# real dataset without touching ingest logic.
COLUMN_MAP: dict[str, dict[str, list[str]]] = {
    "ga4": {
        "date": ["date", "event_date", "ga_date"],
        "source": ["source", "sessionSource", "session_source"],
        "medium": ["medium", "sessionMedium", "session_medium"],
        "campaign": ["campaign", "campaignName", "campaign_name"],
        "revenue": ["revenue", "purchaseRevenue", "totalRevenue"],
        "conversions": ["conversions", "transactions", "purchases"],
        "spend": ["spend", "cost", "adCost"],
    },
    "shopify": {
        "date": ["date", "order_date", "created_at"],
        "revenue": ["revenue", "total_sales", "gross_sales"],
        "orders": ["orders", "order_count"],
    },
}

# source/medium -> normalized channel
CHANNEL_RULES: list[tuple[str, str, str]] = [
    ("google", "cpc", "google"),
    ("bing", "cpc", "microsoft"),
    ("microsoft", "cpc", "microsoft"),
    ("facebook", "paid", "meta"),
    ("meta", "paid", "meta"),
    ("instagram", "paid", "meta"),
]

def get_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the synthetic-data generator and commit sample CSVs**

`scripts/make_sample_data.py`:
```python
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from forecast_core.config import get_rng

CHANNELS = [
    ("google", "cpc", ["brand", "nonbrand", "shopping"]),
    ("bing", "cpc", ["brand", "nonbrand"]),
    ("facebook", "paid", ["prospecting", "retargeting"]),
]

def generate(out_dir: str, n_days: int = 180, seed: int = 42) -> None:
    rng = get_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for source, medium, ctypes in CHANNELS:
        for ctype in ctypes:
            base = rng.uniform(200, 800)
            spend_lvl = rng.uniform(50, 300)
            for i, d in enumerate(dates):
                season = 1 + 0.3 * np.sin(2 * np.pi * i / 7)
                spend = max(0.0, spend_lvl * season * rng.uniform(0.8, 1.2))
                rev = base * season + 3.0 * spend / (1 + spend / 400.0)
                rev = max(0.0, rev * rng.uniform(0.85, 1.15))
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "source": source, "medium": medium,
                    "campaign": f"{source}_{ctype}",
                    "revenue": round(rev, 2),
                    "conversions": int(max(0, rev / 50)),
                    "spend": round(spend, 2),
                })
    ga4 = pd.DataFrame(rows)
    ga4.to_csv(os.path.join(out_dir, "ga4.csv"), index=False)
    shop = (ga4.groupby("date")["revenue"].sum().reset_index()
            .rename(columns={"revenue": "total_sales"}))
    shop["orders"] = (shop["total_sales"] / 60).astype(int)
    shop.to_csv(os.path.join(out_dir, "shopify.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--days", type=int, default=180)
    args = p.parse_args()
    generate(args.out, args.days)
```

`tests/conftest.py`:
```python
import pytest
from scripts.make_sample_data import generate

@pytest.fixture
def sample_data_dir(tmp_path):
    out = tmp_path / "data"
    generate(str(out), n_days=120, seed=7)
    return str(out)
```

Run:
```bash
PYTHONPATH=src python scripts/make_sample_data.py --out data/sample --days 180
git add scripts/make_sample_data.py src/forecast_core/config.py tests/conftest.py tests/test_config.py data/sample/ga4.csv data/sample/shopify.csv
git commit -m "feat: config, deterministic RNG, and committed synthetic sample data"
```
Expected: `data/sample/ga4.csv` and `data/sample/shopify.csv` exist and are tracked.

---

# PHASE 1 — Scored core (Track 1, gating)

## Task 3: Schema-adaptive ingestion

**Files:**
- Create: `src/forecast_core/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `config.COLUMN_MAP`, `config.CHANNEL_RULES`
- Produces:
  - `ingest.load_data(data_dir: str) -> pandas.DataFrame` with canonical long columns: `["date"(datetime64), "channel"(str), "campaign_type"(str), "campaign"(str), "spend"(float), "revenue"(float), "conversions"(float)]`
  - `ingest.resolve_column(df_columns: list[str], candidates: list[str]) -> str | None`
  - `ingest.classify_channel(source: str, medium: str) -> str`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest.py`:
```python
import pandas as pd
from forecast_core import ingest

def test_resolve_column_picks_first_present():
    assert ingest.resolve_column(["a", "Cost"], ["spend", "cost"]) == "Cost"
    assert ingest.resolve_column(["x"], ["spend", "cost"]) is None

def test_classify_channel():
    assert ingest.classify_channel("google", "cpc") == "google"
    assert ingest.classify_channel("bing", "cpc") == "microsoft"
    assert ingest.classify_channel("facebook", "paid") == "meta"
    assert ingest.classify_channel("newsletter", "email") == "other"

def test_load_data_returns_canonical_frame(sample_data_dir):
    df = ingest.load_data(sample_data_dir)
    expected = {"date", "channel", "campaign_type", "campaign",
                "spend", "revenue", "conversions"}
    assert expected.issubset(df.columns)
    assert str(df["date"].dtype).startswith("datetime64")
    assert (df["revenue"] >= 0).all()
    assert set(df["channel"].unique()) <= {"google", "microsoft", "meta", "other"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `ingest.py`**

`src/forecast_core/ingest.py`:
```python
from __future__ import annotations
import glob, os
import pandas as pd
from .config import COLUMN_MAP, CHANNEL_RULES

def resolve_column(df_columns, candidates):
    lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def classify_channel(source: str, medium: str) -> str:
    s, m = str(source).lower(), str(medium).lower()
    for src_key, med_key, channel in CHANNEL_RULES:
        if src_key in s and med_key in m:
            return channel
    return "other"

def _read_any_csv(data_dir: str) -> list[tuple[str, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r}")
    return [(p, pd.read_csv(p)) for p in paths]

def _is_ga4(df: pd.DataFrame) -> bool:
    cm = COLUMN_MAP["ga4"]
    return (resolve_column(df.columns, cm["source"]) is not None
            and resolve_column(df.columns, cm["revenue"]) is not None)

def load_data(data_dir: str) -> pd.DataFrame:
    frames = _read_any_csv(data_dir)
    cm = COLUMN_MAP["ga4"]
    parts = []
    for path, df in frames:
        if not _is_ga4(df):
            continue  # Shopify/other files are used for validation, not the model frame
        date_c = resolve_column(df.columns, cm["date"])
        src_c = resolve_column(df.columns, cm["source"])
        med_c = resolve_column(df.columns, cm["medium"])
        camp_c = resolve_column(df.columns, cm["campaign"])
        rev_c = resolve_column(df.columns, cm["revenue"])
        conv_c = resolve_column(df.columns, cm["conversions"])
        spend_c = resolve_column(df.columns, cm["spend"])
        if date_c is None or rev_c is None:
            raise ValueError(f"{path}: missing required date/revenue columns")
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_c], errors="coerce")
        out["channel"] = [classify_channel(s, m) for s, m in
                          zip(df[src_c] if src_c else "", df[med_c] if med_c else "")]
        out["campaign"] = df[camp_c].astype(str) if camp_c else out["channel"]
        out["campaign_type"] = out["campaign"].str.split("_").str[-1]
        out["revenue"] = pd.to_numeric(df[rev_c], errors="coerce").fillna(0.0)
        out["conversions"] = (pd.to_numeric(df[conv_c], errors="coerce").fillna(0.0)
                              if conv_c else 0.0)
        out["spend"] = (pd.to_numeric(df[spend_c], errors="coerce").fillna(0.0)
                        if spend_c else 0.0)
        parts.append(out)
    if not parts:
        raise ValueError("No GA4-style conversion file detected in data/")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/ingest.py tests/test_ingest.py
git commit -m "feat: schema-adaptive ingestion to canonical long frame"
```

---

## Task 4: Campaign-consistency validation

**Files:**
- Create: `src/forecast_core/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: canonical frame from `ingest.load_data`
- Produces:
  - `validate.ValidationReport` dataclass with fields `n_rows:int`, `n_campaigns:int`, `date_min`, `date_max`, `issues: list[str]`, `ok: bool`
  - `validate.validate_campaigns(df: pandas.DataFrame) -> ValidationReport`

- [ ] **Step 1: Write failing tests**

`tests/test_validate.py`:
```python
import pandas as pd
from forecast_core import ingest, validate

def test_report_ok_on_clean_data(sample_data_dir):
    rep = validate.validate_campaigns(ingest.load_data(sample_data_dir))
    assert rep.ok is True
    assert rep.n_rows > 0
    assert rep.n_campaigns >= 1
    assert rep.issues == []

def test_flags_negative_revenue():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"]),
        "channel": ["google"], "campaign_type": ["brand"],
        "campaign": ["google_brand"], "spend": [10.0],
        "revenue": [-5.0], "conversions": [1.0],
    })
    rep = validate.validate_campaigns(df)
    assert rep.ok is False
    assert any("negative revenue" in s.lower() for s in rep.issues)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_validate.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `validate.py`**

`src/forecast_core/validate.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class ValidationReport:
    n_rows: int
    n_campaigns: int
    date_min: str
    date_max: str
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

def validate_campaigns(df: pd.DataFrame) -> ValidationReport:
    issues: list[str] = []
    required = {"date", "channel", "campaign", "spend", "revenue"}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        issues.append("Input frame is empty after ingestion")
    if "revenue" in df and (df["revenue"] < 0).any():
        issues.append(f"Found {(df['revenue'] < 0).sum()} rows with negative revenue")
    if "spend" in df and (df["spend"] < 0).any():
        issues.append(f"Found {(df['spend'] < 0).sum()} rows with negative spend")
    # campaign consistency: a campaign must map to a single channel
    if {"campaign", "channel"}.issubset(df.columns):
        multi = df.groupby("campaign")["channel"].nunique()
        bad = multi[multi > 1].index.tolist()
        if bad:
            issues.append(f"Campaigns mapped to multiple channels: {bad[:5]}")
    return ValidationReport(
        n_rows=int(len(df)),
        n_campaigns=int(df["campaign"].nunique()) if "campaign" in df else 0,
        date_min=str(df["date"].min()) if "date" in df and not df.empty else "",
        date_max=str(df["date"].max()) if "date" in df and not df.empty else "",
        issues=issues,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_validate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/validate.py tests/test_validate.py
git commit -m "feat: campaign-consistency validation report"
```

---

## Task 5: Feature frame + parquet IO

**Files:**
- Create: `src/forecast_core/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: canonical frame from `ingest.load_data`
- Produces:
  - `features.build_feature_frame(df) -> pandas.DataFrame` aggregated to daily per `(channel, campaign_type, campaign)` with columns `["date","channel","campaign_type","campaign","spend","revenue","conversions","dow","week_index"]`
  - `features.write_features(frame, path)` / `features.read_features(path)` (parquet)

- [ ] **Step 1: Write failing tests**

`tests/test_features.py`:
```python
from forecast_core import ingest, features

def test_build_feature_frame_is_daily_unique(sample_data_dir):
    raw = ingest.load_data(sample_data_dir)
    feats = features.build_feature_frame(raw)
    keys = ["date", "campaign"]
    assert not feats.duplicated(subset=keys).any()
    assert {"dow", "week_index"}.issubset(feats.columns)
    assert feats["dow"].between(0, 6).all()

def test_parquet_roundtrip(sample_data_dir, tmp_path):
    feats = features.build_feature_frame(ingest.load_data(sample_data_dir))
    p = tmp_path / "f.parquet"
    features.write_features(feats, str(p))
    back = features.read_features(str(p))
    assert list(back.columns) == list(feats.columns)
    assert len(back) == len(feats)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `features.py`**

`src/forecast_core/features.py`:
```python
from __future__ import annotations
import pandas as pd

GROUP_KEYS = ["channel", "campaign_type", "campaign"]

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["date"] + GROUP_KEYS, as_index=False)
             .agg(spend=("spend", "sum"),
                  revenue=("revenue", "sum"),
                  conversions=("conversions", "sum")))
    agg = agg.sort_values("date").reset_index(drop=True)
    agg["dow"] = agg["date"].dt.dayofweek
    min_date = agg["date"].min()
    agg["week_index"] = ((agg["date"] - min_date).dt.days // 7).astype(int)
    return agg

def write_features(frame: pd.DataFrame, path: str) -> None:
    frame.to_parquet(path, index=False)

def read_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/features.py tests/test_features.py
git commit -m "feat: daily feature frame with seasonality indices and parquet IO"
```

---

## Task 6: Hill saturation response curve

**Files:**
- Create: `src/forecast_core/response_curves.py`
- Test: `tests/test_response_curves.py`

**Interfaces:**
- Produces:
  - `response_curves.hill(spend, alpha, kappa, slope) -> np.ndarray` — incremental revenue; vectorized over both `spend` and broadcastable params.
  - `response_curves.fit_hill(spend, incremental, rng) -> dict` returning `{"alpha","kappa","slope"}` via least squares (used by train fallback only).

- [ ] **Step 1: Write failing tests**

`tests/test_response_curves.py`:
```python
import numpy as np
from forecast_core import response_curves as rc

def test_hill_monotonic_and_saturating():
    s = np.linspace(0, 1000, 50)
    y = rc.hill(s, alpha=100.0, kappa=200.0, slope=1.5)
    assert np.all(np.diff(y) >= -1e-9)          # monotonic non-decreasing
    assert y[-1] < 100.0 and y[-1] > 50.0        # approaches but below alpha
    assert abs(rc.hill(np.array([0.0]), 100.0, 200.0, 1.5)[0]) < 1e-9

def test_hill_broadcasts_over_param_draws():
    s = np.array([100.0, 200.0])
    alpha = np.array([[100.0], [120.0]])         # (2 draws, 1)
    out = rc.hill(s, alpha, kappa=200.0, slope=1.0)
    assert out.shape == (2, 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_response_curves.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `response_curves.py`**

`src/forecast_core/response_curves.py`:
```python
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def hill(spend, alpha, kappa, slope):
    spend = np.asarray(spend, dtype=float)
    s = np.power(np.clip(spend, 0, None), slope)
    k = np.power(np.asarray(kappa, dtype=float), slope)
    return np.asarray(alpha, dtype=float) * s / (k + s + 1e-12)

def fit_hill(spend, incremental, rng) -> dict:
    spend = np.asarray(spend, float)
    incremental = np.asarray(incremental, float)
    a0 = max(incremental.max(), 1.0) * 1.5
    k0 = max(np.median(spend[spend > 0]) if (spend > 0).any() else 1.0, 1.0)
    x0 = np.array([a0, k0, 1.0])

    def resid(p):
        return hill(spend, p[0], p[1], max(p[2], 0.1)) - incremental

    sol = least_squares(resid, x0, bounds=([1e-6, 1e-6, 0.1], [np.inf, np.inf, 5.0]))
    return {"alpha": float(sol.x[0]), "kappa": float(sol.x[1]), "slope": float(sol.x[2])}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_response_curves.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/response_curves.py tests/test_response_curves.py
git commit -m "feat: Hill saturation response curve (numpy) + LS fit"
```

---

## Task 7: BayesianForecaster — pure-NumPy posterior predictive

**Files:**
- Create: `src/forecast_core/bayesian_predict.py`
- Test: `tests/test_bayesian_predict.py`

**Interfaces:**
- Consumes: `response_curves.hill`, `config.get_rng`
- Produces:
  - `bayesian_predict.CompiledModel` dataclass holding the pickled artifact contents: `series: list[dict]` (each `{"channel","campaign_type","campaign","baseline_draws"(np.ndarray n_draws), "seasonal_dow"(np.ndarray n_draws x 7), "hill"{"alpha","kappa","slope"} each np.ndarray n_draws, "sigma_log"(np.ndarray n_draws), "recent_spend"(float)}`), `n_draws:int`, `last_date`(str ISO), `calibration: dict`
  - `bayesian_predict.BayesianForecaster(model: CompiledModel)` with:
    - `.predict_series(horizon: int, budget_plan: dict | None, rng) -> dict[campaign -> np.ndarray]` returning, per series, an array of shape `(n_draws,)` of total horizon revenue draws, plus a parallel `spend_totals: dict[campaign -> float]`
    - `.save(path)` / classmethod `.load(path) -> BayesianForecaster`

- [ ] **Step 1: Write failing tests (with a hand-built synthetic posterior — no PyMC)**

`tests/test_bayesian_predict.py`:
```python
import numpy as np
from forecast_core.config import get_rng
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster

def _toy_model(n_draws=200):
    rng = get_rng(1)
    series = [{
        "channel": "google", "campaign_type": "brand", "campaign": "google_brand",
        "baseline_draws": rng.normal(300, 10, n_draws),
        "seasonal_dow": rng.normal(0, 1, (n_draws, 7)),
        "hill": {"alpha": rng.normal(150, 5, n_draws),
                 "kappa": np.full(n_draws, 200.0),
                 "slope": np.full(n_draws, 1.2)},
        "sigma_log": np.full(n_draws, 0.05),
        "recent_spend": 100.0,
    }]
    return CompiledModel(series=series, n_draws=n_draws,
                         last_date="2025-06-30", calibration={})

def test_predict_series_shapes_and_horizon_scaling():
    fc = BayesianForecaster(_toy_model())
    out30, sp30 = fc.predict_series(30, None, get_rng(0))
    out90, sp90 = fc.predict_series(90, None, get_rng(0))
    assert out30["google_brand"].shape == (200,)
    # 90-day total revenue should exceed 30-day total revenue (more days)
    assert np.median(out90["google_brand"]) > np.median(out30["google_brand"])
    assert sp90["google_brand"] > sp30["google_brand"]

def test_budget_increase_raises_revenue_but_saturates():
    fc = BayesianForecaster(_toy_model())
    base, _ = fc.predict_series(30, {"google": 100.0}, get_rng(0))
    more, _ = fc.predict_series(30, {"google": 300.0}, get_rng(0))
    assert np.median(more["google_brand"]) > np.median(base["google_brand"])

def test_save_load_roundtrip(tmp_path):
    fc = BayesianForecaster(_toy_model())
    p = tmp_path / "m.pkl"
    fc.save(str(p))
    fc2 = BayesianForecaster.load(str(p))
    a, _ = fc.predict_series(30, None, get_rng(0))
    b, _ = fc2.predict_series(30, None, get_rng(0))
    assert np.allclose(a["google_brand"], b["google_brand"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_bayesian_predict.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `bayesian_predict.py`**

`src/forecast_core/bayesian_predict.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import numpy as np
import pandas as pd
from .response_curves import hill

@dataclass
class CompiledModel:
    series: list[dict]
    n_draws: int
    last_date: str
    calibration: dict = field(default_factory=dict)

class BayesianForecaster:
    def __init__(self, model: CompiledModel):
        self.model = model

    def _future_dows(self, horizon: int) -> np.ndarray:
        start = pd.to_datetime(self.model.last_date) + pd.Timedelta(days=1)
        days = pd.date_range(start, periods=horizon, freq="D")
        return days.dayofweek.to_numpy()

    def predict_series(self, horizon: int, budget_plan, rng):
        dows = self._future_dows(horizon)            # (H,)
        revenue_draws: dict[str, np.ndarray] = {}
        spend_totals: dict[str, float] = {}
        for s in self.model.series:
            nd = self.model.n_draws
            # daily spend over horizon: budget override per channel else run-rate
            if budget_plan and s["channel"] in budget_plan:
                daily_spend = float(budget_plan[s["channel"]])
            else:
                daily_spend = float(s["recent_spend"])
            spend_totals[s["campaign"]] = daily_spend * horizon
            # seasonal component per draw per day: (nd, H)
            seasonal = s["seasonal_dow"][:, dows]
            baseline = s["baseline_draws"][:, None] + seasonal      # (nd, H)
            incr = hill(daily_spend,
                        s["hill"]["alpha"][:, None],
                        s["hill"]["kappa"][:, None],
                        s["hill"]["slope"][:, None])                # (nd, 1)
            mean_daily = np.clip(baseline + incr, 1e-6, None)       # (nd, H)
            # lognormal observation noise, seeded
            sigma = s["sigma_log"][:, None]
            noise = rng.normal(0.0, 1.0, size=mean_daily.shape) * sigma
            daily = mean_daily * np.exp(noise - 0.5 * sigma**2)
            revenue_draws[s["campaign"]] = daily.sum(axis=1)        # (nd,)
        return revenue_draws, spend_totals

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BayesianForecaster":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_bayesian_predict.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/bayesian_predict.py tests/test_bayesian_predict.py
git commit -m "feat: pure-numpy BayesianForecaster posterior-predictive simulation"
```

---

## Task 8: Hierarchical aggregation + correct ROAS

**Files:**
- Create: `src/forecast_core/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: per-series revenue draws + spend totals from `BayesianForecaster.predict_series`, and the series metadata (`channel`, `campaign_type`, `campaign`)
- Produces:
  - `aggregate.aggregate_levels(revenue_draws, spend_totals, series_meta, paid_channels) -> dict` mapping `(level, entity, metric) -> np.ndarray` of draws, where `metric in {"revenue","roas"}`. ROAS draws = `Σrevenue_draws / Σspend` (spend treated as fixed per scenario).

- [ ] **Step 1: Write failing tests**

`tests/test_aggregate.py`:
```python
import numpy as np
from forecast_core import aggregate

def _inputs():
    rev = {"google_brand": np.array([100.0, 120.0]),
           "meta_pros": np.array([40.0, 60.0])}
    spend = {"google_brand": 50.0, "meta_pros": 20.0}
    meta = {"google_brand": {"channel": "google", "campaign_type": "brand"},
            "meta_pros": {"channel": "meta", "campaign_type": "pros"}}
    return rev, spend, meta

def test_total_revenue_is_sum_of_paid_series():
    rev, spend, meta = _inputs()
    out = aggregate.aggregate_levels(rev, spend, meta, ("google", "meta"))
    tot = out[("total", "all", "revenue")]
    assert np.allclose(tot, [140.0, 180.0])

def test_blended_roas_is_ratio_of_sums_not_avg_of_ratios():
    rev, spend, meta = _inputs()
    out = aggregate.aggregate_levels(rev, spend, meta, ("google", "meta"))
    roas = out[("total", "all", "roas")]
    assert np.allclose(roas, [140.0 / 70.0, 180.0 / 70.0])

def test_channel_level_present():
    rev, spend, meta = _inputs()
    out = aggregate.aggregate_levels(rev, spend, meta, ("google", "meta"))
    assert np.allclose(out[("channel", "google", "revenue")], [100.0, 120.0])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_aggregate.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `aggregate.py`**

`src/forecast_core/aggregate.py`:
```python
from __future__ import annotations
import numpy as np

def _accumulate(groups, key, rev, spend):
    r, s = groups.get(key, (None, 0.0))
    r = rev.copy() if r is None else r + rev
    return groups.__setitem__(key, (r, s + spend))

def aggregate_levels(revenue_draws, spend_totals, series_meta, paid_channels):
    groups: dict = {}   # (level, entity) -> (revenue_draws, spend_total)
    paid = set(paid_channels)
    for camp, rev in revenue_draws.items():
        m = series_meta[camp]
        spend = float(spend_totals[camp])
        # campaign level
        _accumulate(groups, ("campaign", camp), rev, spend)
        # campaign_type level
        _accumulate(groups, ("campaign_type", m["campaign_type"]), rev, spend)
        # channel level
        _accumulate(groups, ("channel", m["channel"]), rev, spend)
        # total (paid only, per documented assumption)
        if m["channel"] in paid:
            _accumulate(groups, ("total", "all"), rev, spend)

    out: dict = {}
    for (level, entity), (rev, spend) in groups.items():
        out[(level, entity, "revenue")] = rev
        out[(level, entity, "roas")] = rev / spend if spend > 0 else np.full_like(rev, np.nan)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_aggregate.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/aggregate.py tests/test_aggregate.py
git commit -m "feat: hierarchical aggregation with ratio-of-sums ROAS"
```

---

## Task 9: Uncertainty metrics + conformal recalibration

**Files:**
- Create: `src/forecast_core/uncertainty.py`
- Test: `tests/test_uncertainty.py`

**Interfaces:**
- Produces:
  - `uncertainty.pinball_loss(y_true, q_pred, q) -> float`
  - `uncertainty.wape(y_true, y_pred) -> float`
  - `uncertainty.coverage(y_true, lower, upper) -> float`
  - `uncertainty.apply_conformal(draws: np.ndarray, factor: float) -> np.ndarray` (widens draws around their median by `factor`)

- [ ] **Step 1: Write failing tests**

`tests/test_uncertainty.py`:
```python
import numpy as np
from forecast_core import uncertainty as u

def test_pinball_zero_when_perfect():
    y = np.array([10.0, 20.0])
    assert u.pinball_loss(y, y, 0.5) == 0.0

def test_wape_basic():
    assert abs(u.wape(np.array([100.0]), np.array([90.0])) - 0.1) < 1e-9

def test_coverage_counts_fraction_inside():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 2.0])
    assert u.coverage(y, lo, hi) == 0.5

def test_conformal_widens_spread():
    d = np.array([1.0, 2.0, 3.0])
    w = u.apply_conformal(d, 2.0)
    assert w.std() > d.std()
    assert abs(np.median(w) - np.median(d)) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_uncertainty.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `uncertainty.py`**

`src/forecast_core/uncertainty.py`:
```python
from __future__ import annotations
import numpy as np

def pinball_loss(y_true, q_pred, q: float) -> float:
    y_true, q_pred = np.asarray(y_true, float), np.asarray(q_pred, float)
    diff = y_true - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def wape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else float("nan")

def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, float)
    inside = (y_true >= np.asarray(lower, float)) & (y_true <= np.asarray(upper, float))
    return float(inside.mean())

def apply_conformal(draws: np.ndarray, factor: float) -> np.ndarray:
    med = np.median(draws)
    return med + (np.asarray(draws, float) - med) * float(factor)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_uncertainty.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: pinball/WAPE/coverage metrics and conformal widening"
```

---

## Task 10: Output schema (single source of truth)

**Files:**
- Create: `src/forecast_core/output_schema.py`
- Test: `tests/test_output_schema.py`

**Interfaces:**
- Consumes: the aggregated draws dict from `aggregate.aggregate_levels` (per horizon)
- Produces:
  - `output_schema.COLUMNS = ["level","entity","horizon_days","metric","p10","p50","p90"]`
  - `output_schema.build_predictions(by_horizon: dict[int, dict], quantiles=config.QUANTILES) -> pandas.DataFrame`
  - `output_schema.write_predictions(df, path)` (creates parent dir, fresh write)

- [ ] **Step 1: Write failing tests**

`tests/test_output_schema.py`:
```python
import os, numpy as np, pandas as pd
from forecast_core import output_schema as osch

def _by_horizon():
    draws = {("total", "all", "revenue"): np.arange(100.0),
             ("total", "all", "roas"): np.linspace(1, 3, 100)}
    return {30: draws, 60: draws}

def test_columns_and_rows():
    df = osch.build_predictions(_by_horizon())
    assert list(df.columns) == osch.COLUMNS
    # 2 metrics x 2 horizons = 4 rows for the single (total, all) entity
    assert len(df) == 4
    assert set(df["horizon_days"]) == {30, 60}
    assert (df["p10"] <= df["p50"]).all() and (df["p50"] <= df["p90"]).all()

def test_write_is_fresh(tmp_path):
    p = tmp_path / "out" / "predictions.csv"
    osch.write_predictions(osch.build_predictions(_by_horizon()), str(p))
    osch.write_predictions(osch.build_predictions({30: _by_horizon()[30]}), str(p))
    back = pd.read_csv(p)
    assert set(back["horizon_days"]) == {30}   # overwritten, not appended
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_output_schema.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement `output_schema.py`**

`src/forecast_core/output_schema.py`:
```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .config import QUANTILES

COLUMNS = ["level", "entity", "horizon_days", "metric", "p10", "p50", "p90"]

def build_predictions(by_horizon: dict, quantiles=QUANTILES) -> pd.DataFrame:
    rows = []
    for horizon, draws in by_horizon.items():
        for (level, entity, metric), arr in draws.items():
            arr = np.asarray(arr, float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                p10 = p50 = p90 = float("nan")
            else:
                p10, p50, p90 = np.quantile(arr, quantiles)
            rows.append({"level": level, "entity": entity,
                         "horizon_days": int(horizon), "metric": metric,
                         "p10": p10, "p50": p50, "p90": p90})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["horizon_days", "level", "entity", "metric"]).reset_index(drop=True)

def write_predictions(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_output_schema.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/forecast_core/output_schema.py tests/test_output_schema.py
git commit -m "feat: single-source output schema builder + fresh CSV writer"
```

---

## Task 11: `generate_features.py` CLI

**Files:**
- Create: `src/generate_features.py`
- Test: covered by Task 13 smoke test (CLI integration)

**Interfaces:**
- Consumes: `ingest.load_data`, `validate.validate_campaigns`, `features.build_feature_frame`, `features.write_features`
- Produces: CLI `python src/generate_features.py --data-dir <dir> --out <features.parquet>`; writes parquet; prints validation summary; exits non-zero if validation has blocking issues.

- [ ] **Step 1: Implement the CLI**

`src/generate_features.py`:
```python
from __future__ import annotations
import argparse, sys
from forecast_core import ingest, validate, features

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = ingest.load_data(args.data_dir)
    report = validate.validate_campaigns(df)
    print(f"[validate] rows={report.n_rows} campaigns={report.n_campaigns} "
          f"dates={report.date_min}..{report.date_max} ok={report.ok}")
    for issue in report.issues:
        print(f"[validate] ISSUE: {issue}", file=sys.stderr)
    if not report.ok:
        return 2
    feats = features.build_feature_frame(df)
    features.write_features(feats, args.out)
    print(f"[features] wrote {len(feats)} rows -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it on the sample data**

Run:
```bash
PYTHONPATH=src python src/generate_features.py --data-dir data/sample --out features.parquet
```
Expected: prints `[validate] ... ok=True` and `[features] wrote N rows -> features.parquet`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/generate_features.py
git commit -m "feat: generate_features CLI (ingest+validate+features)"
```

---

## Task 12: `predict.py` CLI

**Files:**
- Create: `src/predict.py`
- Test: `tests/test_predict_cli.py`

**Interfaces:**
- Consumes: `features.read_features`, `BayesianForecaster.load`, `aggregate.aggregate_levels`, `output_schema.build_predictions/write_predictions`, `config.get_rng`, `config.PAID_CHANNELS`, `config.HORIZONS`
- Produces: CLI `python src/predict.py --features <parquet> --model <pkl> --output <csv>`; deterministic; writes predictions for all horizons/levels/metrics.

- [ ] **Step 1: Write failing integration test (builds a tiny pickle, runs CLI)**

`tests/test_predict_cli.py`:
```python
import subprocess, sys, os
import numpy as np, pandas as pd
from forecast_core.config import get_rng
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster
from forecast_core import features, ingest

def _make_model(path):
    rng = get_rng(1); nd = 100
    series = [{
        "channel": "google", "campaign_type": "brand", "campaign": "google_brand",
        "baseline_draws": rng.normal(300, 10, nd),
        "seasonal_dow": rng.normal(0, 1, (nd, 7)),
        "hill": {"alpha": rng.normal(150, 5, nd), "kappa": np.full(nd, 200.0),
                 "slope": np.full(nd, 1.2)},
        "sigma_log": np.full(nd, 0.05), "recent_spend": 100.0,
    }]
    BayesianForecaster(CompiledModel(series, nd, "2025-06-30", {})).save(path)

def test_predict_cli_writes_valid_predictions(tmp_path, sample_data_dir):
    feats = features.build_feature_frame(ingest.load_data(sample_data_dir))
    fpath = tmp_path / "f.parquet"; features.write_features(feats, str(fpath))
    mpath = tmp_path / "m.pkl"; _make_model(str(mpath))
    out = tmp_path / "out.csv"
    env = {**os.environ, "PYTHONPATH": "src"}
    r = subprocess.run([sys.executable, "src/predict.py", "--features", str(fpath),
                        "--model", str(mpath), "--output", str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert list(df.columns) == ["level","entity","horizon_days","metric","p10","p50","p90"]
    assert set(df["horizon_days"]) == {30, 60, 90}
    assert set(df["metric"]) == {"revenue", "roas"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_predict_cli.py -v`
Expected: FAIL (`src/predict.py` does not exist).

- [ ] **Step 3: Implement `predict.py`**

`src/predict.py`:
```python
from __future__ import annotations
import argparse
from forecast_core import features, aggregate, output_schema
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core.config import get_rng, PAID_CHANNELS, HORIZONS

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    _ = features.read_features(args.features)   # validated upstream; reserved for refits
    fc = BayesianForecaster.load(args.model)
    series_meta = {s["campaign"]: {"channel": s["channel"],
                                   "campaign_type": s["campaign_type"]}
                   for s in fc.model.series}

    by_horizon = {}
    for h in HORIZONS:
        rev, spend = fc.predict_series(h, budget_plan=None, rng=get_rng())
        by_horizon[h] = aggregate.aggregate_levels(rev, spend, series_meta, PAID_CHANNELS)

    df = output_schema.build_predictions(by_horizon)
    output_schema.write_predictions(df, args.output)
    print(f"[predict] wrote {len(df)} rows -> {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_predict_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict_cli.py
git commit -m "feat: predict CLI producing probabilistic predictions.csv"
```

---

## Task 13: `run.sh` + clean-pipeline smoke test

**Files:**
- Create: `run.sh`
- Test: `tests/test_run_sh_smoke.py`

**Interfaces:**
- Consumes: `src/generate_features.py`, `src/predict.py`
- Produces: the scored entry point exactly per the submission contract.

- [ ] **Step 1: Write `run.sh`**

`run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

export PYTHONPATH="src:${PYTHONPATH:-}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

python src/generate_features.py --data-dir "$DATA_DIR" --out features.parquet
python src/predict.py --features features.parquet --model "$MODEL_PATH" --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
```

- [ ] **Step 2: Make it executable and write the smoke test**

`tests/test_run_sh_smoke.py`:
```python
import subprocess, os
import pandas as pd

def test_run_sh_end_to_end(tmp_path, sample_data_dir):
    # build a model into a temp pickle using the toy builder used elsewhere
    from tests.test_predict_cli import _make_model
    model = tmp_path / "model.pkl"; _make_model(str(model))
    out = tmp_path / "predictions.csv"
    r = subprocess.run(["bash", "run.sh", sample_data_dir, str(model), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert list(df.columns) == ["level","entity","horizon_days","metric","p10","p50","p90"]
    assert len(df) > 0
```

Run:
```bash
chmod +x run.sh
pytest tests/test_run_sh_smoke.py -v
```
Expected: PASS (1 passed).

- [ ] **Step 3: Run the whole suite**

Run: `pytest -v`
Expected: all tests PASS, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add run.sh tests/test_run_sh_smoke.py
git commit -m "feat: run.sh scored entry point + end-to-end smoke test"
```

---

## Task 14: `train.py` — offline Bayesian fit → compiled pickle

**Files:**
- Create: `train.py`
- Test: manual (produces the committed `pickle/model.pkl`); a lightweight fallback path is unit-tested.

**Interfaces:**
- Consumes: `ingest.load_data`, `features.build_feature_frame`, `response_curves.fit_hill`, `bayesian_predict.CompiledModel/BayesianForecaster`, `config`
- Produces: `pickle/model.pkl`. `train.fit_bayesian(feats, draws, tune) -> CompiledModel` (PyMC). `train.fit_fallback(feats, n_draws, rng) -> CompiledModel` (no PyMC; bootstrap posterior so the pipeline is runnable even before the full fit lands — supports P0).

- [ ] **Step 1: Implement the fallback compiler (P0 unblocker) with a test**

`tests/test_train_fallback.py`:
```python
import numpy as np
from forecast_core import ingest, features
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core.config import get_rng
import train

def test_fallback_builds_usable_model(sample_data_dir):
    feats = features.build_feature_frame(ingest.load_data(sample_data_dir))
    model = train.fit_fallback(feats, n_draws=100, rng=get_rng(0))
    fc = BayesianForecaster(model)
    rev, spend = fc.predict_series(30, None, get_rng(0))
    assert len(rev) >= 1
    for arr in rev.values():
        assert arr.shape == (100,)
        assert np.isfinite(arr).all()
```

`train.py` (fallback + Bayesian + CLI):
```python
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from forecast_core import ingest, features
from forecast_core.response_curves import fit_hill
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster
from forecast_core.config import get_rng, SEED

def _series_table(feats: pd.DataFrame):
    return feats.groupby(["channel", "campaign_type", "campaign"], sort=True)

def fit_fallback(feats: pd.DataFrame, n_draws: int, rng) -> CompiledModel:
    series = []
    last_date = str(feats["date"].max().date())
    for (channel, ctype, camp), g in _series_table(feats):
        daily_rev = g.groupby("date")["revenue"].sum()
        daily_spend = g.groupby("date")["spend"].sum()
        base = float(daily_rev.mean())
        base_sd = float(daily_rev.std() or base * 0.1)
        # seasonal day-of-week deviations from mean
        dow_mean = g.groupby(g["date"].dt.dayofweek)["revenue"].mean()
        seasonal = np.array([float(dow_mean.get(d, base)) - base for d in range(7)])
        hp = fit_hill(daily_spend.to_numpy(),
                      np.clip(daily_rev.to_numpy() - base, 0, None), rng)
        series.append({
            "channel": channel, "campaign_type": ctype, "campaign": camp,
            "baseline_draws": rng.normal(base, max(base_sd, 1.0), n_draws),
            "seasonal_dow": seasonal[None, :] + rng.normal(0, abs(base_sd) * 0.1 + 1e-6,
                                                           (n_draws, 7)),
            "hill": {"alpha": rng.normal(hp["alpha"], hp["alpha"] * 0.1 + 1e-6, n_draws),
                     "kappa": np.full(n_draws, hp["kappa"]),
                     "slope": np.full(n_draws, hp["slope"])},
            "sigma_log": np.full(n_draws, 0.1),
            "recent_spend": float(daily_spend.tail(14).mean() or 0.0),
        })
    return CompiledModel(series=series, n_draws=n_draws, last_date=last_date, calibration={})

def fit_bayesian(feats: pd.DataFrame, draws: int = 1000, tune: int = 1000) -> CompiledModel:
    """Full hierarchical fit. Requires requirements-train.txt (PyMC)."""
    import pymc as pm  # imported lazily so the scored path never needs PyMC
    rng = get_rng(SEED)
    series = []
    last_date = str(feats["date"].max().date())
    for (channel, ctype, camp), g in _series_table(feats):
        d = g.groupby("date").agg(revenue=("revenue", "sum"),
                                  spend=("spend", "sum")).reset_index()
        y = d["revenue"].to_numpy(); spend = d["spend"].to_numpy()
        dow = d["date"].dt.dayofweek.to_numpy()
        with pm.Model():
            base = pm.Normal("base", mu=float(y.mean()), sigma=float(y.std() + 1.0))
            s_dow = pm.Normal("s_dow", 0.0, sigma=float(y.std() + 1.0), shape=7)
            alpha = pm.HalfNormal("alpha", sigma=float(max(y.max(), 1.0)))
            kappa = pm.HalfNormal("kappa", sigma=float(max(np.median(spend[spend > 0])
                                                           if (spend > 0).any() else 1.0, 1.0)))
            slope = pm.TruncatedNormal("slope", mu=1.0, sigma=0.5, lower=0.1, upper=5.0)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            incr = alpha * spend**slope / (kappa**slope + spend**slope + 1e-9)
            mu = pm.math.log(pm.math.clip(base + s_dow[dow] + incr, 1e-6, np.inf))
            pm.Lognormal("obs", mu=mu, sigma=sigma, observed=np.clip(y, 1e-6, None))
            idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                              random_seed=SEED, progressbar=False)
        post = idata.posterior
        flat = lambda name: post[name].to_numpy().reshape(-1, *post[name].shape[2:])
        nd = flat("base").shape[0]
        series.append({
            "channel": channel, "campaign_type": ctype, "campaign": camp,
            "baseline_draws": flat("base"),
            "seasonal_dow": flat("s_dow"),
            "hill": {"alpha": flat("alpha"), "kappa": flat("kappa"), "slope": flat("slope")},
            "sigma_log": flat("sigma"),
            "recent_spend": float(d["spend"].tail(14).mean() or 0.0),
        })
        n_draws = nd
    return CompiledModel(series=series, n_draws=n_draws, last_date=last_date, calibration={})

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sample")
    ap.add_argument("--out", default="pickle/model.pkl")
    ap.add_argument("--method", choices=["bayesian", "fallback"], default="bayesian")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args(argv)
    feats = features.build_feature_frame(ingest.load_data(args.data_dir))
    if args.method == "bayesian":
        model = fit_bayesian(feats, draws=args.draws)
    else:
        model = fit_fallback(feats, n_draws=2000, rng=get_rng(SEED))
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    BayesianForecaster(model).save(args.out)
    print(f"[train] {args.method} model with {len(model.series)} series -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `pytest tests/test_train_fallback.py -v`
Expected: PASS (1 passed).

- [ ] **Step 2: Build the committed model (fallback first so the repo runs out of the box)**

Run:
```bash
PYTHONPATH=src python train.py --data-dir data/sample --out pickle/model.pkl --method fallback
PYTHONPATH=src python -c "from forecast_core.bayesian_predict import BayesianForecaster; print(len(BayesianForecaster.load('pickle/model.pkl').model.series), 'series')"
```
Expected: prints the series count; `pickle/model.pkl` exists.

- [ ] **Step 3: Verify the full pipeline against the committed model**

Run:
```bash
bash run.sh data/sample pickle/model.pkl output/predictions.csv
head output/predictions.csv
```
Expected: `Done. Predictions written to output/predictions.csv`, valid CSV with the 7 columns.

- [ ] **Step 4: Upgrade to the Bayesian fit once data is real**

Run (after obtaining the real dataset into `data/`):
```bash
pip install -r requirements-train.txt
PYTHONPATH=src python train.py --data-dir data --out pickle/model.pkl --method bayesian --draws 1000
bash run.sh data pickle/model.pkl output/predictions.csv
```
Expected: model rebuilt from posterior draws; pipeline still produces valid output.

- [ ] **Step 5: Commit (model + train)**

```bash
git add train.py tests/test_train_fallback.py pickle/model.pkl
git commit -m "feat: offline trainer (Bayesian + fallback) and committed compiled model"
```

> **Verification gate (Track 1 complete):** run `pytest -v` (expect 0 failures) and `bash run.sh data/sample pickle/model.pkl output/predictions.csv` (expect exit 0 + valid CSV) on a fresh `git clone` into a clean venv before moving on. This is the submission's gating contract — do not skip.

---

# PHASE 2 — Backend + Claude causal layer (Track 2)

## Task 15: Pydantic schemas + forecast service wrapping `forecast_core`

**Files:**
- Create: `app/__init__.py`, `app/schemas.py`, `app/services/__init__.py`, `app/services/forecast_service.py`
- Test: `tests/app/test_forecast_service.py`

**Interfaces:**
- Consumes: `BayesianForecaster`, `aggregate`, `output_schema`, `config`
- Produces:
  - `schemas.ForecastRequest{horizon:int, budget_plan:dict[str,float]|None}`, `schemas.SeriesForecast{level,entity,metric,horizon_days,p10,p50,p90}`, `schemas.InsightRequest{diagnostics:dict}`
  - `forecast_service.ForecastService(model_path)` with `.forecast(horizon, budget_plan) -> list[dict]` (rows matching `output_schema.COLUMNS`) and `.diagnostics(horizon, budget_plan) -> dict` (decomposition, saturation status, scenario deltas) for the LLM.

- [ ] **Step 1: Write failing test**

`tests/app/test_forecast_service.py`:
```python
import numpy as np
from tests.test_predict_cli import _make_model
from app.services.forecast_service import ForecastService

def test_service_forecast_and_diagnostics(tmp_path):
    mpath = tmp_path / "m.pkl"; _make_model(str(mpath))
    svc = ForecastService(str(mpath))
    rows = svc.forecast(30, None)
    assert any(r["level"] == "channel" and r["metric"] == "revenue" for r in rows)
    base = svc.forecast(30, {"google": 100.0})
    more = svc.forecast(30, {"google": 400.0})
    g = lambda rs: next(r["p50"] for r in rs if r["level"]=="channel"
                        and r["entity"]=="google" and r["metric"]=="revenue")
    assert g(more) > g(base)
    di: dict = svc.diagnostics(30, None)
    assert "series" in di and "scenario" in di
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/app/test_forecast_service.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement schemas + service**

`app/schemas.py`:
```python
from __future__ import annotations
from pydantic import BaseModel

class ForecastRequest(BaseModel):
    horizon: int = 30
    budget_plan: dict[str, float] | None = None

class SeriesForecast(BaseModel):
    level: str; entity: str; metric: str; horizon_days: int
    p10: float; p50: float; p90: float

class InsightRequest(BaseModel):
    diagnostics: dict
```

`app/services/forecast_service.py`:
```python
from __future__ import annotations
import numpy as np
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core import aggregate, output_schema
from forecast_core.config import get_rng, PAID_CHANNELS

class ForecastService:
    def __init__(self, model_path: str):
        self.fc = BayesianForecaster.load(model_path)
        self.series_meta = {s["campaign"]: {"channel": s["channel"],
                                            "campaign_type": s["campaign_type"]}
                            for s in self.fc.model.series}

    def _agg(self, horizon, budget_plan):
        rev, spend = self.fc.predict_series(horizon, budget_plan, get_rng())
        return aggregate.aggregate_levels(rev, spend, self.series_meta, PAID_CHANNELS)

    def forecast(self, horizon: int, budget_plan) -> list[dict]:
        df = output_schema.build_predictions({horizon: self._agg(horizon, budget_plan)})
        return df.to_dict(orient="records")

    def diagnostics(self, horizon: int, budget_plan) -> dict:
        base = self._agg(horizon, None)
        scen = self._agg(horizon, budget_plan) if budget_plan else base
        def med(d, key): return float(np.nanmedian(d[key])) if key in d else None
        series_status = []
        for s in self.fc.model.series:
            rs = float(s["recent_spend"])
            kappa = float(np.median(s["hill"]["kappa"]))
            series_status.append({
                "channel": s["channel"], "campaign": s["campaign"],
                "recent_spend": rs, "half_saturation": kappa,
                "saturation": "saturated" if rs > kappa else "has_headroom",
            })
        return {
            "horizon_days": horizon,
            "total_revenue_p50": med(base, ("total", "all", "revenue")),
            "blended_roas_p50": med(base, ("total", "all", "roas")),
            "scenario": {
                "applied": bool(budget_plan),
                "budget_plan": budget_plan or {},
                "revenue_p50_base": med(base, ("total", "all", "revenue")),
                "revenue_p50_scenario": med(scen, ("total", "all", "revenue")),
            },
            "series": series_status,
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/app/test_forecast_service.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/schemas.py app/services/ tests/app/test_forecast_service.py
git commit -m "feat: forecast service + diagnostics over forecast_core"
```

---

## Task 16: FastAPI app — `/upload`, `/validate`, `/forecast`, `/simulate`

**Files:**
- Create: `app/main.py`
- Test: `tests/app/test_endpoints.py`

**Interfaces:**
- Consumes: `ForecastService`, `ingest`, `validate`
- Produces: FastAPI `app` with routes. Model path from env `MODEL_PATH` (default `pickle/model.pkl`). Uploaded CSVs saved under a per-process temp dir.

- [ ] **Step 1: Write failing test (TestClient)**

`tests/app/test_endpoints.py`:
```python
import os
from fastapi.testclient import TestClient
from tests.test_predict_cli import _make_model

def _client(tmp_path):
    mpath = tmp_path / "m.pkl"; _make_model(str(mpath))
    os.environ["MODEL_PATH"] = str(mpath)
    from importlib import reload
    import app.main as m; reload(m)
    return TestClient(m.app)

def test_forecast_endpoint(tmp_path):
    c = _client(tmp_path)
    r = c.post("/forecast", json={"horizon": 30})
    assert r.status_code == 200
    assert len(r.json()["rows"]) > 0

def test_simulate_changes_revenue(tmp_path):
    c = _client(tmp_path)
    r = c.post("/simulate", json={"horizon": 30, "budget_plan": {"google": 400.0}})
    assert r.status_code == 200
    assert "diagnostics" in r.json()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/app/test_endpoints.py -v`
Expected: FAIL (`app.main` not defined).

- [ ] **Step 3: Implement `app/main.py`**

`app/main.py`:
```python
from __future__ import annotations
import os, tempfile, shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ForecastRequest
from app.services.forecast_service import ForecastService
from forecast_core import ingest, validate

app = FastAPI(title="AIgnition Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

MODEL_PATH = os.environ.get("MODEL_PATH", "pickle/model.pkl")
_svc = ForecastService(MODEL_PATH)
_UPLOAD_DIR = tempfile.mkdtemp(prefix="aignition_data_")

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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/app/test_endpoints.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/app/test_endpoints.py
git commit -m "feat: FastAPI endpoints for upload/validate/forecast/simulate"
```

---

## Task 17: LLM provider abstraction + Claude insights (offline-testable)

**Files:**
- Create: `app/llm/__init__.py`, `app/llm/provider.py`, `app/llm/claude_insights.py`
- Test: `tests/app/test_insights.py`

**Interfaces:**
- Produces:
  - `provider.LLMProvider` (protocol) with `.complete(system: str, user: str) -> str`
  - `provider.ClaudeProvider(model="claude-opus-4-8")` using the Anthropic SDK (network — demo only)
  - `provider.EchoProvider` (deterministic, offline, for tests)
  - `claude_insights.build_prompt(diagnostics: dict) -> tuple[str, str]` (system, user)
  - `claude_insights.generate_insights(diagnostics: dict, provider: LLMProvider) -> dict{narrative:str, risks:list[str], recommendations:list[str]}`

- [ ] **Step 1: Write failing test (uses EchoProvider; no network)**

`tests/app/test_insights.py`:
```python
from app.llm.provider import EchoProvider
from app.llm import claude_insights as ci

def test_build_prompt_includes_numbers():
    diag = {"horizon_days": 30, "total_revenue_p50": 12345.0,
            "blended_roas_p50": 2.5, "scenario": {"applied": False},
            "series": [{"channel": "google", "saturation": "has_headroom"}]}
    system, user = ci.build_prompt(diag)
    assert "12345" in user and "roas" in user.lower()

def test_generate_insights_offline():
    diag = {"horizon_days": 30, "total_revenue_p50": 100.0, "blended_roas_p50": 2.0,
            "scenario": {"applied": False}, "series": []}
    out = ci.generate_insights(diag, EchoProvider())
    assert set(out) == {"narrative", "risks", "recommendations"}
    assert isinstance(out["risks"], list)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/app/test_insights.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement provider + insights**

`app/llm/provider.py`:
```python
from __future__ import annotations
import json, os
from typing import Protocol

class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...

class EchoProvider:
    """Deterministic offline provider for tests/demos without a key."""
    def complete(self, system: str, user: str) -> str:
        return json.dumps({
            "narrative": "Forecast generated from model diagnostics.",
            "risks": ["Limited history may widen intervals."],
            "recommendations": ["Shift budget toward channels with headroom."],
        })

class ClaudeProvider:
    def __init__(self, model: str = "claude-opus-4-8"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=1200, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
```

`app/llm/claude_insights.py`:
```python
from __future__ import annotations
import json

SYSTEM = (
    "You are a senior e-commerce marketing analyst. You receive structured "
    "forecast diagnostics (already computed by a Bayesian model) and explain "
    "them. Never invent numbers; only interpret what is given. Respond with a "
    "JSON object with keys: narrative (string), risks (array of strings), "
    "recommendations (array of strings)."
)

def build_prompt(diagnostics: dict) -> tuple[str, str]:
    user = (
        "Here are the forecast diagnostics as JSON. Write a causal narrative "
        "explaining the expected revenue and blended ROAS, flag operational "
        "risks, and recommend budget actions based on per-channel saturation.\n\n"
        + json.dumps(diagnostics, indent=2)
    )
    return SYSTEM, user

def generate_insights(diagnostics: dict, provider) -> dict:
    system, user = build_prompt(diagnostics)
    raw = provider.complete(system, user)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"narrative": raw, "risks": [], "recommendations": []}
    return {"narrative": data.get("narrative", ""),
            "risks": list(data.get("risks", [])),
            "recommendations": list(data.get("recommendations", []))}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/app/test_insights.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/llm/ tests/app/test_insights.py
git commit -m "feat: LLM provider abstraction + Claude insights (offline-testable)"
```

---

## Task 18: `/insights` endpoint wiring Claude

**Files:**
- Modify: `app/main.py` (add the route)
- Test: `tests/app/test_insights_endpoint.py`

**Interfaces:**
- Consumes: `ForecastService.diagnostics`, `claude_insights.generate_insights`, provider selection by env (`ANTHROPIC_API_KEY` present → Claude, else Echo).

- [ ] **Step 1: Write failing test**

`tests/app/test_insights_endpoint.py`:
```python
import os
from fastapi.testclient import TestClient
from tests.test_predict_cli import _make_model

def test_insights_endpoint_offline(tmp_path):
    mpath = tmp_path / "m.pkl"; _make_model(str(mpath))
    os.environ["MODEL_PATH"] = str(mpath)
    os.environ.pop("ANTHROPIC_API_KEY", None)   # force EchoProvider
    from importlib import reload
    import app.main as m; reload(m)
    c = TestClient(m.app)
    r = c.post("/insights", json={"horizon": 30})
    assert r.status_code == 200
    assert "narrative" in r.json()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/app/test_insights_endpoint.py -v`
Expected: FAIL (`/insights` 404).

- [ ] **Step 3: Add the route + provider selector to `app/main.py`**

Append to `app/main.py`:
```python
from app.llm.provider import EchoProvider, ClaudeProvider
from app.llm.claude_insights import generate_insights

def _provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeProvider()
    return EchoProvider()

@app.post("/insights")
def insights(req: ForecastRequest):
    diag = _svc.diagnostics(req.horizon, req.budget_plan)
    return generate_insights(diag, _provider())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/app/test_insights_endpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/app/test_insights_endpoint.py
git commit -m "feat: /insights endpoint with Claude/Echo provider selection"
```

---

# PHASE 3 — Frontend (Track 2 demo)

> Frontend tasks use a render/smoke check via the dev server plus Playwright where useful; component code is given in full. Run the API (`uvicorn app.main:app --reload --port 8000`) alongside `npm run dev`.

## Task 19: Next.js scaffold + typed API client

**Files:**
- Create: `frontend/package.json`, `frontend/next.config.mjs`, `frontend/tailwind.config.ts`, `frontend/app/layout.tsx`, `frontend/app/globals.css`, `frontend/lib/api.ts`, `frontend/.env.local`

**Interfaces:**
- Produces: `lib/api.ts` exporting `getForecast`, `simulate`, `getInsights`, `uploadFiles`, `validate` typed against the backend responses.

- [ ] **Step 1: Scaffold**

Run:
```bash
cd frontend
npx create-next-app@14 . --ts --tailwind --app --eslint --src-dir=false --import-alias "@/*" --no-turbo
npm install recharts
```

- [ ] **Step 2: Write the API client**

`frontend/lib/api.ts`:
```typescript
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type ForecastRow = {
  level: string; entity: string; metric: "revenue" | "roas";
  horizon_days: number; p10: number; p50: number; p90: number;
};
export type Diagnostics = {
  horizon_days: number; total_revenue_p50: number; blended_roas_p50: number;
  scenario: { applied: boolean; revenue_p50_base: number; revenue_p50_scenario: number };
  series: { channel: string; campaign: string; recent_spend: number;
            half_saturation: number; saturation: string }[];
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

export const getForecast = (horizon: number) =>
  post<{ rows: ForecastRow[] }>("/forecast", { horizon });
export const simulate = (horizon: number, budget_plan: Record<string, number>) =>
  post<{ rows: ForecastRow[]; diagnostics: Diagnostics }>("/simulate", { horizon, budget_plan });
export const getInsights = (horizon: number, budget_plan?: Record<string, number>) =>
  post<{ narrative: string; risks: string[]; recommendations: string[] }>(
    "/insights", { horizon, budget_plan });
export async function validate() {
  const r = await fetch(`${BASE}/validate`); return r.json();
}
export async function uploadFiles(files: File[]) {
  const fd = new FormData(); files.forEach((f) => fd.append("files", f));
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return r.json();
}
```

`frontend/.env.local`:
```
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: build completes, exit 0.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/lib/api.ts frontend/.env.local frontend/next.config.mjs
git commit -m "feat: Next.js scaffold + typed API client"
```

---

## Task 20: Upload & validation screen

**Files:**
- Create: `frontend/app/page.tsx`, `frontend/components/UploadPanel.tsx`

**Interfaces:**
- Consumes: `uploadFiles`, `validate`

- [ ] **Step 1: Implement the component**

`frontend/components/UploadPanel.tsx`:
```tsx
"use client";
import { useState } from "react";
import { uploadFiles, validate } from "@/lib/api";

export default function UploadPanel() {
  const [report, setReport] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    setBusy(true);
    await uploadFiles(Array.from(e.target.files));
    setReport(await validate());
    setBusy(false);
  }
  return (
    <section className="rounded-lg border p-4">
      <h2 className="text-lg font-medium">1 · Upload data</h2>
      <input type="file" multiple accept=".csv" onChange={onUpload} className="mt-2" />
      {busy && <p className="text-sm text-gray-500">Validating…</p>}
      {report && (
        <div className="mt-3 text-sm">
          <p>Rows: {report.n_rows} · Campaigns: {report.n_campaigns}</p>
          <p>Dates: {report.date_min} → {report.date_max}</p>
          <p className={report.ok ? "text-green-700" : "text-red-700"}>
            {report.ok ? "Validation passed" : `Issues: ${report.issues.join("; ")}`}
          </p>
        </div>
      )}
    </section>
  );
}
```

`frontend/app/page.tsx`:
```tsx
import UploadPanel from "@/components/UploadPanel";
import ForecastDashboard from "@/components/ForecastDashboard";
import BudgetSimulator from "@/components/BudgetSimulator";
import InsightsPanel from "@/components/InsightsPanel";

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">AIgnition · Revenue & ROAS forecasting</h1>
      <UploadPanel />
      <ForecastDashboard />
      <BudgetSimulator />
      <InsightsPanel />
    </main>
  );
}
```

- [ ] **Step 2: Verify the page renders (with API running)**

Run:
```bash
uvicorn app.main:app --port 8000 &   # from repo root, separate shell
cd frontend && npm run dev
```
Expected: visiting `http://localhost:3000` shows the upload panel; uploading `data/sample/*.csv` shows a validation summary.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx frontend/components/UploadPanel.tsx
git commit -m "feat: upload + validation screen"
```

---

## Task 21: Forecast dashboard with P10–P90 fan charts

**Files:**
- Create: `frontend/components/ForecastDashboard.tsx`, `frontend/components/FanChart.tsx`

**Interfaces:**
- Consumes: `getForecast`, `ForecastRow`

- [ ] **Step 1: Implement the fan chart + dashboard**

`frontend/components/FanChart.tsx`:
```tsx
"use client";
import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export type FanPoint = { label: string; p10: number; p50: number; p90: number };

export default function FanChart({ data }: { data: FanPoint[] }) {
  const shaped = data.map((d) => ({ ...d, band: [d.p10, d.p90] as [number, number] }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={shaped}>
        <XAxis dataKey="label" /><YAxis /><Tooltip />
        <Area dataKey="band" stroke="none" fill="#bfdbfe" />
        <Line dataKey="p50" stroke="#2563eb" dot={false} strokeWidth={2} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
```

`frontend/components/ForecastDashboard.tsx`:
```tsx
"use client";
import { useEffect, useState } from "react";
import { getForecast, ForecastRow } from "@/lib/api";
import FanChart, { FanPoint } from "./FanChart";

const HORIZONS = [30, 60, 90];

export default function ForecastDashboard() {
  const [horizon, setHorizon] = useState(30);
  const [rows, setRows] = useState<ForecastRow[]>([]);
  useEffect(() => { getForecast(horizon).then((r) => setRows(r.rows)); }, [horizon]);

  const channelRevenue: FanPoint[] = rows
    .filter((r) => r.level === "channel" && r.metric === "revenue")
    .map((r) => ({ label: r.entity, p10: r.p10, p50: r.p50, p90: r.p90 }));
  const totalRoas = rows.find((r) => r.level === "total" && r.metric === "roas");

  return (
    <section className="rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">2 · Forecast</h2>
        <div className="flex gap-2">
          {HORIZONS.map((h) => (
            <button key={h} onClick={() => setHorizon(h)}
              className={`rounded px-3 py-1 text-sm ${h === horizon ? "bg-blue-600 text-white" : "bg-gray-100"}`}>
              {h}d
            </button>
          ))}
        </div>
      </div>
      {totalRoas && (
        <p className="mt-2 text-sm">Blended ROAS (P50): <b>{totalRoas.p50.toFixed(2)}</b>
          {" "}(P10 {totalRoas.p10.toFixed(2)} – P90 {totalRoas.p90.toFixed(2)})</p>
      )}
      <FanChart data={channelRevenue} />
    </section>
  );
}
```

- [ ] **Step 2: Verify build + render**

Run: `cd frontend && npm run build`
Expected: build exit 0; dashboard shows per-channel revenue band and blended ROAS range.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/ForecastDashboard.tsx frontend/components/FanChart.tsx
git commit -m "feat: forecast dashboard with P10-P90 fan chart and horizon toggle"
```

---

## Task 22: Budget simulator

**Files:**
- Create: `frontend/components/BudgetSimulator.tsx`

**Interfaces:**
- Consumes: `simulate`, `Diagnostics`

- [ ] **Step 1: Implement the simulator**

`frontend/components/BudgetSimulator.tsx`:
```tsx
"use client";
import { useState } from "react";
import { simulate, Diagnostics } from "@/lib/api";

const CHANNELS = ["google", "microsoft", "meta"];

export default function BudgetSimulator() {
  const [budget, setBudget] = useState<Record<string, number>>(
    { google: 100, microsoft: 100, meta: 100 });
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  async function run() { setDiag((await simulate(30, budget)).diagnostics); }
  return (
    <section className="rounded-lg border p-4">
      <h2 className="text-lg font-medium">3 · Budget simulator</h2>
      {CHANNELS.map((c) => (
        <div key={c} className="mt-2 flex items-center gap-3">
          <span className="w-24 text-sm capitalize">{c}</span>
          <input type="range" min={0} max={1000} value={budget[c]}
            onChange={(e) => setBudget({ ...budget, [c]: Number(e.target.value) })} />
          <span className="w-16 text-sm">${budget[c]}/day</span>
        </div>
      ))}
      <button onClick={run} className="mt-3 rounded bg-blue-600 px-3 py-1 text-sm text-white">
        Re-forecast
      </button>
      {diag && (
        <p className="mt-3 text-sm">
          Revenue P50: base ${diag.scenario.revenue_p50_base?.toFixed(0)} →
          scenario ${diag.scenario.revenue_p50_scenario?.toFixed(0)}
        </p>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Verify build + interaction**

Run: `cd frontend && npm run build`
Expected: build exit 0; moving sliders + Re-forecast updates the base→scenario revenue line.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/BudgetSimulator.tsx
git commit -m "feat: per-channel budget simulator with scenario delta"
```

---

## Task 23: AI insights panel

**Files:**
- Create: `frontend/components/InsightsPanel.tsx`

**Interfaces:**
- Consumes: `getInsights`

- [ ] **Step 1: Implement the panel**

`frontend/components/InsightsPanel.tsx`:
```tsx
"use client";
import { useState } from "react";
import { getInsights } from "@/lib/api";

export default function InsightsPanel() {
  const [data, setData] = useState<{ narrative: string; risks: string[];
    recommendations: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  async function explain() { setBusy(true); setData(await getInsights(30)); setBusy(false); }
  return (
    <section className="rounded-lg border p-4">
      <h2 className="text-lg font-medium">4 · AI insights</h2>
      <button onClick={explain} className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm text-white">
        {busy ? "Thinking…" : "Explain this forecast"}
      </button>
      {data && (
        <div className="mt-3 space-y-2 text-sm">
          <p>{data.narrative}</p>
          {data.risks.length > 0 && (
            <div><b>Risks</b><ul className="list-disc pl-5">
              {data.risks.map((r, i) => <li key={i}>{r}</li>)}</ul></div>)}
          {data.recommendations.length > 0 && (
            <div><b>Recommendations</b><ul className="list-disc pl-5">
              {data.recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul></div>)}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Verify build + render**

Run: `cd frontend && npm run build`
Expected: build exit 0; "Explain this forecast" populates narrative/risks/recommendations (Echo provider works offline; Claude when `ANTHROPIC_API_KEY` is set on the backend).

- [ ] **Step 3: Commit**

```bash
git add frontend/components/InsightsPanel.tsx
git commit -m "feat: AI insights panel (Claude-backed narrative/risks/recs)"
```

---

# PHASE 4 — Documentation & submission packaging

## Task 24: Technical documentation + architecture overview

**Files:**
- Create: `docs/technical-documentation.md`, `docs/architecture.md`

**Interfaces:** none (prose deliverables required by the brief).

- [ ] **Step 1: Write `docs/technical-documentation.md`** covering exactly the brief's required headings, each with concrete content drawn from the implementation:
  - forecasting methodology (hierarchical Bayesian media-response model; offline-compile trick)
  - model selection (why Bayesian vs quantile GBM/A; partial pooling rationale)
  - data preprocessing logic (ingest normalization, channel classification, daily aggregation, seasonality indices)
  - assumptions (paid-channel total; spend basis; placeholder output schema)
  - limitations (no adstock by default; attribution taken as-is; data-history dependence)
  - AI integration strategy (Claude consumes diagnostics only; grounded narrative)

- [ ] **Step 2: Write `docs/architecture.md`** covering the four required headings — frontend stack, backend stack, forecasting pipeline, LLM integration workflow — and embed the two-track diagram description (inputs → train.py → model.pkl → forecast_core → {run.sh→predictions.csv} / {FastAPI+Next.js→Claude}).

- [ ] **Step 3: Verify the brief's deliverable headings are all present**

Run:
```bash
grep -Ei "methodology|model selection|preprocessing|assumption|limitation|AI integration" docs/technical-documentation.md
grep -Ei "frontend|backend|forecasting pipeline|LLM" docs/architecture.md
```
Expected: each required heading matches at least once.

- [ ] **Step 4: Commit**

```bash
git add docs/technical-documentation.md docs/architecture.md
git commit -m "docs: technical documentation and architecture overview"
```

---

## Task 25: README, demo workflow, and final clean-clone verification

**Files:**
- Create: `README.md`, `docs/demo-workflow.md`

**Interfaces:** none.

- [ ] **Step 1: Write `README.md`** with: one-paragraph description; Python version (3.11); the exact run command (`./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`); how to install (`pip install -r requirements.txt`); how to run the demo app (`requirements-app.txt`, uvicorn, `frontend` npm); and a statement that the scored path needs no internet.

- [ ] **Step 2: Write `docs/demo-workflow.md`** as a 4-step walkthrough mapping to the brief: data ingestion → forecast generation → budget simulation → AI-generated business insights, with the exact UI clicks and the screenshots to capture for the presentation.

- [ ] **Step 3: Final clean-clone verification (the submission gate)**

Run:
```bash
cd /tmp && rm -rf aignition-verify && git clone <your-local-or-remote-repo> aignition-verify
cd aignition-verify
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt && pip install -e .
bash run.sh ./data/sample ./pickle/model.pkl ./output/predictions.csv
test -s output/predictions.csv && echo "OUTPUT OK"
```
Expected: ends with `Done. Predictions written to ...` and `OUTPUT OK`. No network used, no manual fixes.

- [ ] **Step 4: Run the full test suite one final time**

Run: `pytest -v`
Expected: all PASS, 0 failures.

- [ ] **Step 5: Commit + push public repo**

```bash
git add README.md docs/demo-workflow.md
git commit -m "docs: README + demo workflow; final verification"
git branch -M main
git remote add origin <public-github-url>
git push -u origin main
```

> **Submission action items (not code):** confirm output format vs the official launch spec (adjust `output_schema.py` if needed); set repo to public; email repo URL + exact command + team name/members/college to sunitha.k@netelixir.us before 2026-07-19 22:00 IST.

---

## Self-review

**1. Spec coverage** — every spec section maps to at least one task:

| Spec § | Requirement | Task(s) |
|---|---|---|
| §3/§7 | Probabilistic revenue + ROAS, 30/60/90, levels | T7, T8, T10, T12 |
| §7 | Seasonality | T5 (dow/week), T14 (seasonal_dow) |
| §7 | Saturation / budget conditioning | T6, T7, T15 |
| §6 | Offline-Bayesian compile trick | T7, T14 |
| §9 | run.sh contract, schema-adaptive ingest, validation | T3, T4, T11, T13 |
| §8 | Single-source output schema | T10 |
| §10 | FastAPI + Claude causal layer | T15–T18 |
| §11 | Next.js demo (4 screens) | T19–T23 |
| §13 | Reproducibility, seeds, pinned deps, smoke test | T1, T2, T13, T25 |
| §14 | Deliverables (docs, architecture, demo) | T24, T25 |
| Guide #41 | model committed, no LFS/.gitignore trap | T1 (.gitignore), T14 (commit pkl), T25 (clone test) |
| Guide #46 | output to OUTPUT_PATH, announced format, fresh write | T10, T12, T13 |

**2. Placeholder scan** — no "TBD/TODO/implement later"; every code step contains runnable code; the only deferred items are the explicitly-flagged external action items (official output format, public repo, team info), which are not code.

**3. Type consistency** — verified the names threaded across tasks: `BayesianForecaster.predict_series(horizon, budget_plan, rng) -> (revenue_draws: dict, spend_totals: dict)` (T7) is consumed identically in T8, T12, T15; `aggregate.aggregate_levels(revenue_draws, spend_totals, series_meta, paid_channels) -> dict[(level,entity,metric)->ndarray]` (T8) is consumed in T10/T12/T15; `output_schema.COLUMNS` (T10) is asserted in T12/T13; `ForecastService.diagnostics` (T15) shape is consumed by `claude_insights.build_prompt` (T17) and the `/insights` route (T18). `_make_model` test helper (T12) is reused by T13/T15/T16/T18.
