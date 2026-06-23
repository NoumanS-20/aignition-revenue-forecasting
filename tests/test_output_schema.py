import numpy as np
import pandas as pd
from forecast_core import output_schema as osch


def _by_horizon():
    draws = {("total", "all", "revenue"): np.arange(100.0),
             ("total", "all", "roas"): np.linspace(1, 3, 100)}
    return {30: draws, 60: draws}


def test_columns_and_rows():
    df = osch.build_predictions(_by_horizon())
    assert list(df.columns) == osch.COLUMNS
    # 2 metrics x 2 horizons = 4 rows for the single (total, all) entity
    assert len(df) == 4
    assert set(df["horizon_days"]) == {30, 60}
    assert (df["p10"] <= df["p50"]).all() and (df["p50"] <= df["p90"]).all()


def test_write_is_fresh(tmp_path):
    p = tmp_path / "out" / "predictions.csv"
    osch.write_predictions(osch.build_predictions(_by_horizon()), str(p))
    osch.write_predictions(osch.build_predictions({30: _by_horizon()[30]}), str(p))
    back = pd.read_csv(p)
    assert set(back["horizon_days"]) == {30}   # overwritten, not appended
