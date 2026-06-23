from forecast_core import config


def test_constants_present():
    assert config.SEED == 42
    assert config.HORIZONS == (30, 60, 90)
    assert config.LEVELS == ("total", "channel", "campaign_type", "campaign")
    assert config.QUANTILES == (0.10, 0.50, 0.90)


def test_rng_is_deterministic():
    a = config.get_rng().normal(size=5)
    b = config.get_rng().normal(size=5)
    assert (a == b).all()
