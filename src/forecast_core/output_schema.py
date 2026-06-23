from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .config import QUANTILES

COLUMNS = ["level", "entity", "horizon_days", "metric", "p10", "p50", "p90"]


def build_predictions(by_horizon: dict, quantiles=QUANTILES) -> pd.DataFrame:
    rows = []
    for horizon, draws in by_horizon.items():
        for (level, entity, metric), arr in draws.items():
            arr = np.asarray(arr, float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                p10 = p50 = p90 = float("nan")
            else:
                p10, p50, p90 = np.quantile(arr, quantiles)
            rows.append({"level": level, "entity": entity,
                         "horizon_days": int(horizon), "metric": metric,
                         "p10": p10, "p50": p50, "p90": p90})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return (df.sort_values(["horizon_days", "level", "entity", "metric"])
              .reset_index(drop=True))


def write_predictions(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
