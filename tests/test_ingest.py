from forecast_core import ingest


def test_resolve_column_picks_first_present():
    assert ingest.resolve_column(["a", "Cost"], ["spend", "cost"]) == "Cost"
    assert ingest.resolve_column(["x"], ["spend", "cost"]) is None


def test_classify_channel():
    assert ingest.classify_channel("google", "cpc") == "google"
    assert ingest.classify_channel("bing", "cpc") == "microsoft"
    assert ingest.classify_channel("facebook", "paid") == "meta"
    assert ingest.classify_channel("newsletter", "email") == "other"


def test_load_data_returns_canonical_frame(sample_data_dir):
    df = ingest.load_data(sample_data_dir)
    expected = {"date", "channel", "campaign_type", "campaign",
                "spend", "revenue", "conversions"}
    assert expected.issubset(df.columns)
    assert str(df["date"].dtype).startswith("datetime64")
    assert (df["revenue"] >= 0).all()
    assert set(df["channel"].unique()) <= {"google", "microsoft", "meta", "other"}
