import numpy as np
from forecast_core import ingest, features
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core.config import get_rng
import train


def test_fallback_builds_usable_model(sample_data_dir):
    feats = features.build_feature_frame(ingest.load_data(sample_data_dir))
    model = train.fit_fallback(feats, n_draws=100, rng=get_rng(0))
    assert len(model.groups) >= 1
    fc = BayesianForecaster(model)
    rev, spend, meta = fc.predict_from_features(feats, 30, None, get_rng(0))
    assert len(rev) >= 1
    for arr in rev.values():
        assert arr.shape == (100,)
        assert np.isfinite(arr).all()
