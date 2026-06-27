from __future__ import annotations
import numpy as np

SEED: int = 42
PAID_CHANNELS: tuple[str, ...] = ("google", "microsoft", "meta")
HORIZONS: tuple[int, ...] = (30, 60, 90)
LEVELS: tuple[str, ...] = ("total", "channel", "campaign_type", "campaign")
METRICS: tuple[str, ...] = ("revenue", "roas")
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# All monetary values are USD (per organizer clarification).
CURRENCY: str = "USD"

# Candidate source-column names for ad-platform exports
# (Google Ads, Microsoft/Bing Ads, Meta Ads). Per the organizer Q&A the inputs
# are the ad platforms' spend/performance data -- GA4 and Shopify are NOT used.
# Candidates include the exact names in the provided dataset (Google uses
# metrics_*; Bing uses TitleCase; Meta uses date_start/spend/conversion).
# Editing this dict adapts ingestion to a new schema without touching logic.
COLUMN_MAP: dict[str, list[str]] = {
    "date": ["date", "day", "report_date", "week", "month",
             "segments_date", "date_start", "timeperiod", "time_period"],
    "channel": ["channel", "platform", "source", "account", "ad_network", "network"],
    "campaign": ["campaign", "campaign_name", "campaignname"],
    "campaign_type": ["campaign_type", "campaigntype", "type",
                      "advertising_channel_type", "campaign_advertising_channel_type"],
    "spend": ["spend", "cost", "amount_spent", "ad_spend", "media_cost",
              "metrics_cost_micros", "cost_micros"],
    # Revenue / conversion value. 'conversion' is last so real revenue columns win;
    # for Meta (no revenue column) it resolves to the 'conversion' value column.
    "revenue": ["revenue", "conv_value", "conversion_value", "conversions_value",
                "metrics_conversions_value", "purchase_value", "total_revenue",
                "sales", "conv_val", "conversion"],
    "conversions": ["conversions", "metrics_conversions", "purchases",
                    "transactions", "orders"],
}

# Spend columns reported in micros (1e-6 units), e.g. Google Ads cost.
MICROS_COLUMNS: set[str] = {"metrics_cost_micros", "cost_micros"}

# Map free-text platform/source tokens to a normalized channel.
CHANNEL_ALIASES: dict[str, str] = {
    "google": "google", "google ads": "google", "googleads": "google",
    "adwords": "google",
    "bing": "microsoft", "microsoft": "microsoft", "msft": "microsoft",
    "microsoft ads": "microsoft",
    "meta": "meta", "facebook": "meta", "fb": "meta", "instagram": "meta",
    "meta ads": "meta",
}

# Infer channel from the file name when there is no channel column
# (ad-platform exports are commonly one file per platform).
FILENAME_CHANNEL_HINTS: list[tuple[str, str]] = [
    ("google", "google"), ("adwords", "google"),
    ("bing", "microsoft"), ("microsoft", "microsoft"), ("msft", "microsoft"),
    ("meta", "meta"), ("facebook", "meta"), ("instagram", "meta"), ("fb", "meta"),
]


def get_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
