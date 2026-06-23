import numpy as np
from forecast_core import uncertainty as u


def test_pinball_zero_when_perfect():
    y = np.array([10.0, 20.0])
    assert u.pinball_loss(y, y, 0.5) == 0.0


def test_wape_basic():
    assert abs(u.wape(np.array([100.0]), np.array([90.0])) - 0.1) < 1e-9


def test_coverage_counts_fraction_inside():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 2.0])
    assert u.coverage(y, lo, hi) == 0.5


def test_conformal_widens_spread():
    d = np.array([1.0, 2.0, 3.0])
    w = u.apply_conformal(d, 2.0)
    assert w.std() > d.std()
    assert abs(np.median(w) - np.median(d)) < 1e-9
