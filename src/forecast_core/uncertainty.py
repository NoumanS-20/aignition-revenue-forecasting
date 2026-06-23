from __future__ import annotations
import numpy as np


def pinball_loss(y_true, q_pred, q: float) -> float:
    y_true, q_pred = np.asarray(y_true, float), np.asarray(q_pred, float)
    diff = y_true - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def wape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else float("nan")


def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, float)
    inside = (y_true >= np.asarray(lower, float)) & (y_true <= np.asarray(upper, float))
    return float(inside.mean())


def apply_conformal(draws: np.ndarray, factor: float) -> np.ndarray:
    med = np.median(draws)
    return med + (np.asarray(draws, float) - med) * float(factor)
