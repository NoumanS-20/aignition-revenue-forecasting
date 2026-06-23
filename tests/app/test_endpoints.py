import os
from fastapi.testclient import TestClient
from tests.test_predict_cli import _make_model


def _client(tmp_path):
    mpath = tmp_path / "m.pkl"
    _make_model(str(mpath))
    os.environ["MODEL_PATH"] = str(mpath)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from importlib import reload
    import app.main as m
    reload(m)
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


def test_insights_endpoint_offline(tmp_path):
    c = _client(tmp_path)
    r = c.post("/insights", json={"horizon": 30})
    assert r.status_code == 200
    assert "narrative" in r.json()
