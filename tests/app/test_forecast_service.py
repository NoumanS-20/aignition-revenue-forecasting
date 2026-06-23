from tests.test_predict_cli import _make_model
from app.services.forecast_service import ForecastService


def test_service_forecast_and_diagnostics(tmp_path):
    mpath = tmp_path / "m.pkl"
    _make_model(str(mpath))
    svc = ForecastService(str(mpath))
    rows = svc.forecast(30, None)
    assert any(r["level"] == "channel" and r["metric"] == "revenue" for r in rows)
    base = svc.forecast(30, {"google": 100.0})
    more = svc.forecast(30, {"google": 400.0})

    def g(rs):
        return next(r["p50"] for r in rs if r["level"] == "channel"
                    and r["entity"] == "google" and r["metric"] == "revenue")

    assert g(more) > g(base)
    di = svc.diagnostics(30, None)
    assert "series" in di and "scenario" in di
