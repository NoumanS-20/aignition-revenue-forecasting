from __future__ import annotations
import glob
import os
import pandas as pd
from .config import COLUMN_MAP, CHANNEL_RULES


def resolve_column(df_columns, candidates):
    lower = {str(c).lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def classify_channel(source: str, medium: str) -> str:
    s, m = str(source).lower(), str(medium).lower()
    for src_key, med_key, channel in CHANNEL_RULES:
        if src_key in s and med_key in m:
            return channel
    return "other"


def _read_any_csv(data_dir: str) -> list[tuple[str, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r}")
    return [(p, pd.read_csv(p)) for p in paths]


def _is_ga4(df: pd.DataFrame) -> bool:
    cm = COLUMN_MAP["ga4"]
    return (resolve_column(df.columns, cm["source"]) is not None
            and resolve_column(df.columns, cm["revenue"]) is not None)


def load_data(data_dir: str) -> pd.DataFrame:
    frames = _read_any_csv(data_dir)
    cm = COLUMN_MAP["ga4"]
    parts = []
    for path, df in frames:
        if not _is_ga4(df):
            continue  # Shopify/other files are used for validation, not the model frame
        date_c = resolve_column(df.columns, cm["date"])
        src_c = resolve_column(df.columns, cm["source"])
        med_c = resolve_column(df.columns, cm["medium"])
        camp_c = resolve_column(df.columns, cm["campaign"])
        rev_c = resolve_column(df.columns, cm["revenue"])
        conv_c = resolve_column(df.columns, cm["conversions"])
        spend_c = resolve_column(df.columns, cm["spend"])
        if date_c is None or rev_c is None:
            raise ValueError(f"{path}: missing required date/revenue columns")
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_c], errors="coerce")
        src_vals = df[src_c] if src_c else pd.Series([""] * len(df))
        med_vals = df[med_c] if med_c else pd.Series([""] * len(df))
        out["channel"] = [classify_channel(s, m) for s, m in zip(src_vals, med_vals)]
        out["campaign"] = df[camp_c].astype(str) if camp_c else out["channel"]
        out["campaign_type"] = out["campaign"].str.split("_").str[-1]
        out["revenue"] = pd.to_numeric(df[rev_c], errors="coerce").fillna(0.0)
        out["conversions"] = (pd.to_numeric(df[conv_c], errors="coerce").fillna(0.0)
                              if conv_c else 0.0)
        out["spend"] = (pd.to_numeric(df[spend_c], errors="coerce").fillna(0.0)
                        if spend_c else 0.0)
        parts.append(out)
    if not parts:
        raise ValueError("No GA4-style conversion file detected in data/")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)
