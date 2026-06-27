from __future__ import annotations
import numpy as np
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core import ingest, features, aggregate, output_schema
from forecast_core.config import get_rng, PAID_CHANNELS


class ForecastService:
    def __init__(self, model_path: str, data_dir: str = "data/sample"):
        self.fc = BayesianForecaster.load(model_path)
        self.paid = getattr(self.fc.model, "paid_channels", PAID_CHANNELS)
        self.set_data_dir(data_dir)

    def set_data_dir(self, data_dir: str) -> None:
        """(Re)load the features the forecasts run on (e.g. after an upload)."""
        self.feats = features.build_feature_frame(ingest.load_data(data_dir))

    def _agg(self, horizon, budget_plan):
        rev, spend, meta = self.fc.predict_from_features(
            self.feats, horizon, budget_plan, get_rng())
        return aggregate.aggregate_levels(rev, spend, meta, self.paid)

    def forecast(self, horizon: int, budget_plan) -> list[dict]:
        df = output_schema.build_predictions({horizon: self._agg(horizon, budget_plan)})
        return df.to_dict(orient="records")

    def diagnostics(self, horizon: int, budget_plan) -> dict:
        base = self._agg(horizon, None)
        scen = self._agg(horizon, budget_plan) if budget_plan else base

        def med(d, key):
            return float(np.nanmedian(d[key])) if key in d else None

        # per-channel recent spend + revenue p50 (drives the causal narrative)
        recent = self.feats.copy()
        chan_spend = recent.groupby("channel")["spend"].sum().to_dict()
        series_status = []
        for ch in sorted(set(self.feats["channel"])):
            series_status.append({
                "channel": ch,
                "recent_spend": float(chan_spend.get(ch, 0.0)),
                "revenue_p50": med(base, ("channel", ch, "revenue")),
                "roas_p50": med(base, ("channel", ch, "roas")),
            })
        return {
            "horizon_days": horizon,
            "total_revenue_p50": med(base, ("total", "all", "revenue")),
            "blended_roas_p50": med(base, ("total", "all", "roas")),
            "scenario": {
                "applied": bool(budget_plan),
                "budget_plan": budget_plan or {},
                "revenue_p50_base": med(base, ("total", "all", "revenue")),
                "revenue_p50_scenario": med(scen, ("total", "all", "revenue")),
            },
            "series": series_status,
        }
