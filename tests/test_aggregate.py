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
