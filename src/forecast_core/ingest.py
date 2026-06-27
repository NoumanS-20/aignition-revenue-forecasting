from __future__ import annotations
import glob
import os
import pandas as pd
from .config import COLUMN_MAP, MICROS_COLUMNS, CHANNEL_ALIASES, FILENAME_CHANNEL_HINTS


def normalize_campaign_type(value: str) -> str:
    """Normalize campaign-type labels so platforms align (e.g. PerformanceMax,
    PERFORMANCE_MAX -> performance_max; SEARCH/Search -> search)."""
    t = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    t = t.replace("performancemax", "performance_max")
    return t or "all"


def resolve_column(df_columns, candidates):
    """Return the actual column name matching the first candidate present (case-insensitive)."""
    lower = {str(c).lower().strip(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_channel(value: str) -> str:
    """Map a free-text platform/source value to google/microsoft/meta/other."""
    v = str(value).lower().strip()
    if v in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[v]
    for token, channel in CHANNEL_ALIASES.items():
        if token in v:
            return channel
    return "other"


def channel_from_filename(path: str) -> str:
    """Infer the channel from the file name (one-file-per-platform exports)."""
    name = os.path.basename(path).lower()
    for token, channel in FILENAME_CHANNEL_HINTS:
        if token in name:
            return channel
    return "other"


def _read_any_csv(data_dir: str) -> list[tuple[str, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r}")
    return [(p, pd.read_csv(p)) for p in paths]


def _normalize_frame(path: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize one ad-platform CSV to the canonical long frame, or None if it isn't one.

    Robust to extra columns (ignored) and missing optional columns (filled).
    """
    date_c = resolve_column(df.columns, COLUMN_MAP["date"])
    rev_c = resolve_column(df.columns, COLUMN_MAP["revenue"])
    spend_c = resolve_column(df.columns, COLUMN_MAP["spend"])
    # Need a date plus at least one monetary signal to be a performance file.
    if date_c is None or (rev_c is None and spend_c is None):
        return None

    chan_c = resolve_column(df.columns, COLUMN_MAP["channel"])
    camp_c = resolve_column(df.columns, COLUMN_MAP["campaign"])
    ctype_c = resolve_column(df.columns, COLUMN_MAP["campaign_type"])
    conv_c = resolve_column(df.columns, COLUMN_MAP["conversions"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_c], errors="coerce")
    if chan_c is not None:
        out["channel"] = df[chan_c].map(normalize_channel)
    else:
        out["channel"] = channel_from_filename(path)
    out["campaign"] = df[camp_c].astype(str) if camp_c is not None else out["channel"]
    if ctype_c is not None:
        out["campaign_type"] = df[ctype_c].map(normalize_campaign_type)
    else:
        # No campaign-type column (e.g. Meta): derive from the campaign name token.
        out["campaign_type"] = (out["campaign"].str.split("_").str[0]
                                .map(normalize_campaign_type))
    out["revenue"] = (pd.to_numeric(df[rev_c], errors="coerce").fillna(0.0)
                      if rev_c is not None else 0.0)
    spend = (pd.to_numeric(df[spend_c], errors="coerce").fillna(0.0)
             if spend_c is not None else 0.0)
    if spend_c is not None and str(spend_c).lower() in MICROS_COLUMNS:
        spend = spend / 1_000_000.0   # micros -> currency units
    out["spend"] = spend
    out["conversions"] = (pd.to_numeric(df[conv_c], errors="coerce").fillna(0.0)
                          if conv_c is not None else 0.0)
    return out


def load_data(data_dir: str) -> pd.DataFrame:
    """Read all ad-platform CSVs under data_dir into one canonical long frame.

    Columns: date, channel, campaign_type, campaign, spend, revenue, conversions.
    """
    frames = _read_any_csv(data_dir)
    parts = []
    for path, df in frames:
        norm = _normalize_frame(path, df)
        if norm is not None:
            parts.append(norm)
    if not parts:
        raise ValueError(
            "No ad-platform performance CSVs detected in data/ "
            "(each file needs a date column plus a revenue or spend column)."
        )
    out = pd.concat(parts, ignore_index=True).dropna(subset=["date"])
    return out.sort_values("date").reset_index(drop=True)
