from __future__ import annotations
import numpy as np
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core import aggregate, output_schema
from forecast_core.config import get_rng, PAID_CHANNELS


class ForecastService:
    def __init__(self, model_path: str):
        self.fc = BayesianForecaster.load(model_path)
        self.series_meta = {s["campaign"]: {"channel": s["channel"],
                                            "campaign_type": s["campaign_type"]}
                            for s in self.fc.model.series}

    def _agg(self, horizon, budget_plan):
        rev, spend = self.fc.predict_series(horizon, budget_plan, get_rng())
        return aggregate.aggregate_levels(rev, spend, self.series_meta, PAID_CHANNELS)

    def forecast(self, horizon: int, budget_plan) -> list[dict]:
        df = output_schema.build_predictions({horizon: self._agg(horizon, budget_plan)})
        return df.to_dict(orient="records")

    def diagnostics(self, horizon: int, budget_plan) -> dict:
        base = self._agg(horizon, None)
        scen = self._agg(horizon, budget_plan) if budget_plan else base

        def med(d, key):
            return float(np.nanmedian(d[key])) if key in d else None

        series_status = []
        for s in self.fc.model.series:
            rs = float(s["recent_spend"])
            kappa = float(np.median(s["hill"]["kappa"]))
            series_status.append({
                "channel": s["channel"], "campaign": s["campaign"],
                "recent_spend": rs, "half_saturation": kappa,
                "saturation": "saturated" if rs > kappa else "has_headroom",
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
