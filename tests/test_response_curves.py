import numpy as np
from forecast_core import response_curves as rc


def test_hill_monotonic_and_saturating():
    s = np.linspace(0, 1000, 50)
    y = rc.hill(s, alpha=100.0, kappa=200.0, slope=1.5)
    assert np.all(np.diff(y) >= -1e-9)          # monotonic non-decreasing
    assert y[-1] < 100.0 and y[-1] > 50.0        # approaches but below alpha
    assert abs(rc.hill(np.array([0.0]), 100.0, 200.0, 1.5)[0]) < 1e-9


def test_hill_broadcasts_over_param_draws():
    s = np.array([100.0, 200.0])
    alpha = np.array([[100.0], [120.0]])         # (2 draws, 1)
    out = rc.hill(s, alpha, kappa=200.0, slope=1.0)
    assert out.shape == (2, 2)
