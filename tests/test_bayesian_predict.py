import numpy as np
import pandas as pd
from forecast_core.config import get_rng
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster


def _feats():
    dates = pd.date_range("2025-06-01", periods=28, freq="D")
    rows = [{"date": d, "channel": "google", "campaign_type": "brand",
             "campaign": "brand_us", "spend": 100.0, "revenue": 300.0,
             "conversions": 3} for d in dates]
    return pd.DataFrame(rows)


def _model(nd=200):
    rng = get_rng(1)
    g = {"seasonal_mult": np.ones((nd, 7)) * np.exp(rng.normal(0, 0.01, (nd, 7))),
         "kappa_rel": np.full(nd, 2.0), "slope": np.full(nd, 1.0),
         "sigma_log": np.full(nd, 0.05)}
    return CompiledModel(groups={}, channel_groups={}, global_group=g,
                         n_draws=nd, last_date="2025-06-28")


def test_predict_shapes_and_horizon_scaling():
    fc = BayesianForecaster(_model())
    f = _feats()
    r30, s30, m30 = fc.predict_from_features(f, 30, None, get_rng(0))
    r90, _, _ = fc.predict_from_features(f, 90, None, get_rng(0))
    sid = "google::brand_us"
    assert r30[sid].shape == (200,)
    assert m30[sid]["channel"] == "google"
    assert np.median(r90[sid]) > np.median(r30[sid])  # more days -> larger total


def test_budget_increase_raises_revenue_but_saturates():
    fc = BayesianForecaster(_model())
    f = _feats()
    base, _, _ = fc.predict_from_features(f, 30, {"google": 100.0}, get_rng(0))
    more, _, _ = fc.predict_from_features(f, 30, {"google": 400.0}, get_rng(0))
    sid = "google::brand_us"
    assert np.median(more[sid]) > np.median(base[sid])


def test_save_load_roundtrip(tmp_path):
    fc = BayesianForecaster(_model())
    f = _feats()
    p = tmp_path / "m.pkl"
    fc.save(str(p))
    fc2 = BayesianForecaster.load(str(p))
    a, _, _ = fc.predict_from_features(f, 30, None, get_rng(0))
    b, _, _ = fc2.predict_from_features(f, 30, None, get_rng(0))
    assert np.allclose(a["google::brand_us"], b["google::brand_us"])
