from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from forecast_core.config import get_rng

# One file per ad platform (channel inferred from the file name), each with
# columns: date, campaign, campaign_type, spend, revenue, conversions. All USD.
PLATFORMS = [
    ("google_ads.csv", ["brand", "nonbrand", "shopping"]),
    ("microsoft_ads.csv", ["brand", "nonbrand"]),
    ("meta_ads.csv", ["prospecting", "retargeting"]),
]


def generate(out_dir: str, n_days: int = 180, seed: int = 42) -> None:
    rng = get_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    for fname, ctypes in PLATFORMS:
        rows = []
        for ctype in ctypes:
            base = rng.uniform(200, 800)
            spend_lvl = rng.uniform(50, 300)
            for i, d in enumerate(dates):
                season = 1 + 0.3 * np.sin(2 * np.pi * i / 7)
                spend = max(0.0, spend_lvl * season * rng.uniform(0.8, 1.2))
                rev = base * season + 3.0 * spend / (1 + spend / 400.0)
                rev = max(0.0, rev * rng.uniform(0.85, 1.15))
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "campaign": f"{ctype}_us",
                    "campaign_type": ctype,
                    "spend": round(spend, 2),
                    "revenue": round(rev, 2),
                    "conversions": int(max(0, rev / 50)),
                })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--days", type=int, default=180)
    args = p.parse_args()
    generate(args.out, args.days)
