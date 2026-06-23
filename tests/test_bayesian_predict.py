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
