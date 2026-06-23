from __future__ import annotations
from pydantic import BaseModel


class ForecastRequest(BaseModel):
    horizon: int = 30
    budget_plan: dict[str, float] | None = None


class SeriesForecast(BaseModel):
    level: str
    entity: str
    metric: str
    horizon_days: int
    p10: float
    p50: float
    p90: float


class InsightRequest(BaseModel):
    diagnostics: dict
