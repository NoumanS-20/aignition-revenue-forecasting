import subprocess
import os
import shutil
import pandas as pd
import pytest
from tests.test_predict_cli import _make_model, REPO_ROOT


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_run_sh_end_to_end(tmp_path, sample_data_dir):
    model = tmp_path / "model.pkl"
    _make_model(str(model))
    out = tmp_path / "predictions.csv"
    r = subprocess.run(["bash", "run.sh", sample_data_dir, str(model), str(out)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert list(df.columns) == ["level", "entity", "horizon_days", "metric",
                                "p10", "p50", "p90"]
    assert len(df) > 0
