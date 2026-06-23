import subprocess
import sys
import os
import numpy as np
import pandas as pd
from forecast_core.config import get_rng
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster
from forecast_core import features, ingest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_model(path):
    rng = get_rng(1)
    nd = 100
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
    fpath = tmp_path / "f.parquet"
    features.write_features(feats, str(fpath))
    mpath = tmp_path / "m.pkl"
    _make_model(str(mpath))
    out = tmp_path / "out.csv"
    r = subprocess.run([sys.executable, "src/predict.py", "--features", str(fpath),
                        "--model", str(mpath), "--output", str(out)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert list(df.columns) == ["level", "entity", "horizon_days", "metric",
                                "p10", "p50", "p90"]
    assert set(df["horizon_days"]) == {30, 60, 90}
    assert set(df["metric"]) == {"revenue", "roas"}
