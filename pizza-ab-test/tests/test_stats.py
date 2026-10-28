import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from pizza_ab import stats as s


def test_sample_size_matches_statsmodels():
    for p, mde in [(0.115, 0.05), (0.30, 0.03), (0.05, 0.10)]:
        ours = s.sample_size_two_proportions(p, mde)
        theirs = NormalIndPower().solve_power(proportion_effectsize(p * (1 + mde), p), alpha=0.05, power=0.8)
        assert ours == pytest.approx(theirs, rel=0.02)


def test_power_is_consistent_with_sample_size():
    n = s.sample_size_two_proportions(0.115, 0.05)
    assert s.power_two_proportions(n, 0.115, 0.05) == pytest.approx(0.80, abs=0.005)
    assert s.mde_for_sample_size(n, 0.115) == pytest.approx(0.05, abs=0.001)


def test_mean_test_matches_statsmodels_unequal_var():
    rng = np.random.default_rng(0)
    a, b = rng.binomial(1, 0.11, 40_000), rng.binomial(1, 0.12, 40_000)
    r = s.mean_test(a, b)
    z, p = CompareMeans(DescrStatsW(b), DescrStatsW(a)).ztest_ind(usevar="unequal")
    assert r.p_value == pytest.approx(p, rel=1e-6)
    assert r.diff_ci[0] < r.diff < r.diff_ci[1]


def test_ratio_test_reduces_to_mean_test_with_unit_denominators():
    rng = np.random.default_rng(1)
    a, b = rng.poisson(0.3, 20_000), rng.poisson(0.33, 20_000)
    r1 = s.ratio_test(a, np.ones_like(a), b, np.ones_like(b))
    r2 = s.mean_test(a, b)
    assert r1.rel_lift == pytest.approx(r2.rel_lift)
    assert r1.p_value == pytest.approx(r2.p_value, rel=1e-3)


def test_cuped_reduces_variance_and_keeps_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(size=50_000)
    y = 0.6 * x + rng.normal(size=50_000)
    y_adj, theta, red = s.cuped(y, x)
    assert theta == pytest.approx(0.6, abs=0.02)
    assert y_adj.mean() == pytest.approx(y.mean())
    assert red == pytest.approx(0.36 / 1.36, abs=0.01)  # rho^2


def test_srm():
    assert s.srm_test(50_000, 50_100) > 0.5
    assert s.srm_test(50_000, 51_000) < 0.01


def test_holm_matches_statsmodels():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.002])
    assert np.allclose(s.holm(p), multipletests(p, method="holm")[1])


def test_obrien_fleming_controls_error():
    bounds = s.obrien_fleming_bounds(14, n_sim=100_000)
    assert bounds[0] > bounds[-1]
    assert 1.96 < bounds[-1] < 2.2
    rng = np.random.default_rng(3)
    k = np.arange(1, 15)
    z = np.cumsum(rng.standard_normal((50_000, 14)), axis=1) / np.sqrt(k)
    assert (np.abs(z) > bounds).any(axis=1).mean() == pytest.approx(0.05, abs=0.005)


def test_msprt_is_monotone_and_valid_under_null():
    rng = np.random.default_rng(4)
    n_sims, looks, n = 2_000, 50, 1_000
    rejections = 0
    for _ in range(n_sims):
        c = np.cumsum(rng.binomial(n, 0.1, looks)); t = np.cumsum(rng.binomial(n, 0.1, looks))
        m = n * np.arange(1, looks + 1)
        pc, pt = c / m, t / m
        p = s.msprt_always_valid_p(pt - pc, pc * (1 - pc) / m + pt * (1 - pt) / m, tau=0.01)
        assert np.all(np.diff(p) <= 1e-12)
        rejections += p[-1] < 0.05
    assert rejections / n_sims <= 0.05
