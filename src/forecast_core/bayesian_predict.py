from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import numpy as np
import pandas as pd
from .response_curves import hill
from .config import PAID_CHANNELS

RECENT_DAYS = 28  # window used to anchor each campaign's baseline/run-rate


@dataclass
class CompiledModel:
    """Transferable, relative parameters learned offline.

    `groups` maps (channel, campaign_type) -> param dict with draw arrays:
      seasonal_mult (n_draws, 7)  day-of-week multipliers centered ~1
      kappa_rel     (n_draws,)    Hill half-saturation as a multiple of run-rate
      slope         (n_draws,)    Hill slope
      sigma_log     (n_draws,)    log-normal noise scale
    `channel_groups` and `global_group` are the same structure for fallback.
    """
    groups: dict
    channel_groups: dict
    global_group: dict
    n_draws: int
    last_date: str
    paid_channels: tuple = PAID_CHANNELS
    calibration: dict = field(default_factory=dict)
    version: int = 2


class BayesianForecaster:
    def __init__(self, model: CompiledModel):
        self.model = model

    def _future_dows(self, last_date, horizon: int) -> np.ndarray:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        return pd.date_range(start, periods=horizon, freq="D").dayofweek.to_numpy()

    def _params_for(self, channel: str, campaign_type: str) -> dict:
        m = self.model
        return (m.groups.get((channel, campaign_type))
                or m.channel_groups.get(channel)
                or m.global_group)

    def predict_from_features(self, feats: pd.DataFrame, horizon: int,
                              budget_plan=None, rng=None):
        """Forecast every campaign present in `feats` over `horizon` days.

        Each campaign's baseline (mean daily revenue) and run-rate spend come from
        its own recent history; the response/seasonality shapes come from the
        learned (channel, campaign_type) group (with channel/global fallback).
        Returns (revenue_draws, spend_totals, series_meta) keyed by channel::campaign.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        feats = feats.copy()
        feats["date"] = pd.to_datetime(feats["date"])
        last_date = feats["date"].max()
        dows = self._future_dows(last_date, horizon)
        recent = feats[feats["date"] >= last_date - pd.Timedelta(days=RECENT_DAYS - 1)]
        stats = (recent.groupby(["channel", "campaign_type", "campaign"], sort=True)
                 .agg(mu=("revenue", "mean"), s0=("spend", "mean")).reset_index())
        chan_spend = stats.groupby("channel")["s0"].sum().to_dict()
        chan_count = stats.groupby("channel")["campaign"].count().to_dict()

        eps = 1e-9
        revenue_draws, spend_totals, series_meta = {}, {}, {}
        for r in stats.itertuples(index=False):
            ch, ct, camp = r.channel, r.campaign_type, r.campaign
            mu = float(r.mu) if np.isfinite(r.mu) else 0.0
            s0 = float(r.s0) if np.isfinite(r.s0) else 0.0
            sid = f"{ch}::{camp}"
            p = self._params_for(ch, ct)

            if budget_plan and ch in budget_plan:
                tot = float(chan_spend.get(ch, 0.0))
                if tot > 0:
                    new_s = float(budget_plan[ch]) * s0 / tot
                else:
                    new_s = float(budget_plan[ch]) / max(int(chan_count.get(ch, 1)), 1)
            else:
                new_s = s0

            kappa = p["kappa_rel"] * max(s0, eps)        # (nd,)
            slope = p["slope"]                            # (nd,)
            resp = hill(new_s, 1.0, kappa, slope) / (hill(s0, 1.0, kappa, slope) + eps)
            seas = p["seasonal_mult"][:, dows]            # (nd, H)
            mean_daily = np.clip(mu * seas * resp[:, None], 0.0, None)
            sigma = p["sigma_log"][:, None]
            noise = rng.normal(0.0, 1.0, size=mean_daily.shape) * sigma
            daily = mean_daily * np.exp(noise - 0.5 * sigma ** 2)
            revenue_draws[sid] = daily.sum(axis=1)
            spend_totals[sid] = new_s * horizon
            series_meta[sid] = {"channel": ch, "campaign_type": ct, "campaign": camp}
        return revenue_draws, spend_totals, series_meta

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BayesianForecaster":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
