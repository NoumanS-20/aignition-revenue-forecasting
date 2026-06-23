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
