import pandas as pd
from forecast_core import ingest


def test_resolve_column_picks_first_present():
    assert ingest.resolve_column(["a", "Cost"], ["spend", "cost"]) == "Cost"
    assert ingest.resolve_column(["x"], ["spend", "cost"]) is None


def test_normalize_channel():
    assert ingest.normalize_channel("Google Ads") == "google"
    assert ingest.normalize_channel("bing") == "microsoft"
    assert ingest.normalize_channel("Facebook") == "meta"
    assert ingest.normalize_channel("tiktok") == "other"


def test_channel_from_filename():
    assert ingest.channel_from_filename("/x/google_ads.csv") == "google"
    assert ingest.channel_from_filename("/x/meta_ads.csv") == "meta"
    assert ingest.channel_from_filename("/x/bing_ads.csv") == "microsoft"


def test_load_data_returns_canonical_frame(sample_data_dir):
    df = ingest.load_data(sample_data_dir)
    expected = {"date", "channel", "campaign_type", "campaign",
                "spend", "revenue", "conversions"}
    assert expected.issubset(df.columns)
    assert str(df["date"].dtype).startswith("datetime64")
    assert (df["revenue"] >= 0).all()
    assert set(df["channel"].unique()) <= {"google", "microsoft", "meta", "other"}


def test_robust_to_extra_and_missing_columns(tmp_path):
    # 'cost' should resolve to spend, extra column ignored, missing conversions -> 0.
    p = tmp_path / "google_ads.csv"
    pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "campaign": ["brand_us", "brand_us"],
        "campaign_type": ["brand", "brand"],
        "cost": [10.0, 12.0],
        "revenue": [50.0, 55.0],
        "some_extra_metric": ["a", "b"],
    }).to_csv(p, index=False)
    df = ingest.load_data(str(tmp_path))
    assert df["channel"].iloc[0] == "google"
    assert df["spend"].iloc[0] == 10.0
    assert (df["conversions"] == 0.0).all()


def test_explicit_channel_column_is_used(tmp_path):
    p = tmp_path / "performance.csv"  # name gives no channel hint
    pd.DataFrame({
        "date": ["2025-01-01"], "platform": ["Microsoft Ads"],
        "campaign": ["brand_us"], "spend": [5.0], "revenue": [20.0],
    }).to_csv(p, index=False)
    df = ingest.load_data(str(tmp_path))
    assert df["channel"].iloc[0] == "microsoft"
