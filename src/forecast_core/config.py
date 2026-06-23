from __future__ import annotations
import numpy as np

SEED: int = 42
PAID_CHANNELS: tuple[str, ...] = ("google", "microsoft", "meta")
HORIZONS: tuple[int, ...] = (30, 60, 90)
LEVELS: tuple[str, ...] = ("total", "channel", "campaign_type", "campaign")
METRICS: tuple[str, ...] = ("revenue", "roas")
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Assumed source-column names. Editing this dict adapts ingestion to the
# real dataset without touching ingest logic.
COLUMN_MAP: dict[str, dict[str, list[str]]] = {
    "ga4": {
        "date": ["date", "event_date", "ga_date"],
        "source": ["source", "sessionSource", "session_source"],
        "medium": ["medium", "sessionMedium", "session_medium"],
        "campaign": ["campaign", "campaignName", "campaign_name"],
        "revenue": ["revenue", "purchaseRevenue", "totalRevenue"],
        "conversions": ["conversions", "transactions", "purchases"],
        "spend": ["spend", "cost", "adCost"],
    },
    "shopify": {
        "date": ["date", "order_date", "created_at"],
        "revenue": ["revenue", "total_sales", "gross_sales"],
        "orders": ["orders", "order_count"],
    },
}

# source/medium -> normalized channel
CHANNEL_RULES: list[tuple[str, str, str]] = [
    ("google", "cpc", "google"),
    ("bing", "cpc", "microsoft"),
    ("microsoft", "cpc", "microsoft"),
    ("facebook", "paid", "meta"),
    ("meta", "paid", "meta"),
    ("instagram", "paid", "meta"),
]


def get_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
