from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationReport:
    n_rows: int
    n_campaigns: int
    date_min: str
    date_max: str
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def validate_campaigns(df: pd.DataFrame) -> ValidationReport:
    issues: list[str] = []
    required = {"date", "channel", "campaign", "spend", "revenue"}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        issues.append("Input frame is empty after ingestion")
    if "revenue" in df and (df["revenue"] < 0).any():
        issues.append(f"Found {(df['revenue'] < 0).sum()} rows with negative revenue")
    if "spend" in df and (df["spend"] < 0).any():
        issues.append(f"Found {(df['spend'] < 0).sum()} rows with negative spend")
    # campaign consistency: a campaign must map to a single channel
    if {"campaign", "channel"}.issubset(df.columns):
        multi = df.groupby("campaign")["channel"].nunique()
        bad = multi[multi > 1].index.tolist()
        if bad:
            issues.append(f"Campaigns mapped to multiple channels: {bad[:5]}")
    return ValidationReport(
        n_rows=int(len(df)),
        n_campaigns=int(df["campaign"].nunique()) if "campaign" in df else 0,
        date_min=str(df["date"].min()) if "date" in df and not df.empty else "",
        date_max=str(df["date"].max()) if "date" in df and not df.empty else "",
        issues=issues,
    )
