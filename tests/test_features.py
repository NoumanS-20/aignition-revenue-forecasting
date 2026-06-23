from forecast_core import ingest, features


def test_build_feature_frame_is_daily_unique(sample_data_dir):
    raw = ingest.load_data(sample_data_dir)
    feats = features.build_feature_frame(raw)
    keys = ["date", "campaign"]
    assert not feats.duplicated(subset=keys).any()
    assert {"dow", "week_index"}.issubset(feats.columns)
    assert feats["dow"].between(0, 6).all()


def test_parquet_roundtrip(sample_data_dir, tmp_path):
    feats = features.build_feature_frame(ingest.load_data(sample_data_dir))
    p = tmp_path / "f.parquet"
    features.write_features(feats, str(p))
    back = features.read_features(str(p))
    assert list(back.columns) == list(feats.columns)
    assert len(back) == len(feats)
