from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares


def hill(spend, alpha, kappa, slope):
    spend = np.asarray(spend, dtype=float)
    s = np.power(np.clip(spend, 0, None), slope)
    k = np.power(np.asarray(kappa, dtype=float), slope)
    return np.asarray(alpha, dtype=float) * s / (k + s + 1e-12)


def fit_hill(spend, incremental, rng) -> dict:
    spend = np.asarray(spend, float)
    incremental = np.asarray(incremental, float)
    a0 = max(incremental.max(), 1.0) * 1.5
    k0 = max(np.median(spend[spend > 0]) if (spend > 0).any() else 1.0, 1.0)
    x0 = np.array([a0, k0, 1.0])

    def resid(p):
        return hill(spend, p[0], p[1], max(p[2], 0.1)) - incremental

    sol = least_squares(resid, x0, bounds=([1e-6, 1e-6, 0.1], [np.inf, np.inf, 5.0]))
    return {"alpha": float(sol.x[0]), "kappa": float(sol.x[1]), "slope": float(sol.x[2])}
