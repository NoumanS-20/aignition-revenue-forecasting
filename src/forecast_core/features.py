from __future__ import annotations
import pandas as pd

GROUP_KEYS = ["channel", "campaign_type", "campaign"]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["date"] + GROUP_KEYS, as_index=False)
             .agg(spend=("spend", "sum"),
                  revenue=("revenue", "sum"),
                  conversions=("conversions", "sum")))
    agg = agg.sort_values("date").reset_index(drop=True)
    agg["dow"] = agg["date"].dt.dayofweek
    min_date = agg["date"].min()
    agg["week_index"] = ((agg["date"] - min_date).dt.days // 7).astype(int)
    return agg


def write_features(frame: pd.DataFrame, path: str) -> None:
    frame.to_parquet(path, index=False)


def read_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
