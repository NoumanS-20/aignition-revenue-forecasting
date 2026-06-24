from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import numpy as np
import pandas as pd
from .response_curves import hill


@dataclass
class CompiledModel:
    series: list[dict]
    n_draws: int
    last_date: str
    calibration: dict = field(default_factory=dict)


class BayesianForecaster:
    def __init__(self, model: CompiledModel):
        self.model = model

    def _future_dows(self, horizon: int) -> np.ndarray:
        start = pd.to_datetime(self.model.last_date) + pd.Timedelta(days=1)
        days = pd.date_range(start, periods=horizon, freq="D")
        return days.dayofweek.to_numpy()

    def predict_series(self, horizon: int, budget_plan, rng):
        dows = self._future_dows(horizon)            # (H,)
        revenue_draws: dict[str, np.ndarray] = {}
        spend_totals: dict[str, float] = {}
        for s in self.model.series:
            # Unique series id: campaign names can repeat across channels.
            sid = s.get("series_id") or f'{s["channel"]}::{s["campaign"]}'
            # daily spend over horizon: budget override per channel else run-rate
            if budget_plan and s["channel"] in budget_plan:
                daily_spend = float(budget_plan[s["channel"]])
            else:
                daily_spend = float(s["recent_spend"])
            spend_totals[sid] = daily_spend * horizon
            # seasonal component per draw per day: (nd, H)
            seasonal = s["seasonal_dow"][:, dows]
            baseline = s["baseline_draws"][:, None] + seasonal      # (nd, H)
            incr = hill(daily_spend,
                        s["hill"]["alpha"][:, None],
                        s["hill"]["kappa"][:, None],
                        s["hill"]["slope"][:, None])                # (nd, 1)
            mean_daily = np.clip(baseline + incr, 1e-6, None)       # (nd, H)
            # lognormal observation noise, seeded
            sigma = s["sigma_log"][:, None]
            noise = rng.normal(0.0, 1.0, size=mean_daily.shape) * sigma
            daily = mean_daily * np.exp(noise - 0.5 * sigma**2)
            revenue_draws[sid] = daily.sum(axis=1)                  # (nd,)
        return revenue_draws, spend_totals

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BayesianForecaster":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
