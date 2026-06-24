import numpy as np
from forecast_core import aggregate


def _inputs():
    # keyed by unique series id "channel::campaign"
    rev = {"google::brand_us": np.array([100.0, 120.0]),
           "meta::pros_us": np.array([40.0, 60.0])}
    spend = {"google::brand_us": 50.0, "meta::pros_us": 20.0}
    meta = {"google::brand_us": {"channel": "google", "campaign_type": "brand",
                                 "campaign": "brand_us"},
            "meta::pros_us": {"channel": "meta", "campaign_type": "pros",
                              "campaign": "pros_us"}}
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


def test_campaign_entity_is_channel_qualified():
    rev, spend, meta = _inputs()
    out = aggregate.aggregate_levels(rev, spend, meta, ("google", "meta"))
    assert ("campaign", "google:brand_us", "revenue") in out


def test_same_campaign_name_across_channels_does_not_merge():
    rev = {"google::brand_us": np.array([100.0]), "microsoft::brand_us": np.array([30.0])}
    spend = {"google::brand_us": 50.0, "microsoft::brand_us": 10.0}
    meta = {"google::brand_us": {"channel": "google", "campaign_type": "brand",
                                 "campaign": "brand_us"},
            "microsoft::brand_us": {"channel": "microsoft", "campaign_type": "brand",
                                    "campaign": "brand_us"}}
    out = aggregate.aggregate_levels(rev, spend, meta, ("google", "microsoft"))
    assert np.allclose(out[("campaign", "google:brand_us", "revenue")], [100.0])
    assert np.allclose(out[("campaign", "microsoft:brand_us", "revenue")], [30.0])
    # campaign_type "brand" aggregates across both channels
    assert np.allclose(out[("campaign_type", "brand", "revenue")], [130.0])
