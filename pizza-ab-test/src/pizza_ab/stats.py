"""Statistics for the experiment: power, tests, variance reduction, sequential methods.

Everything is implemented from first principles (normal approximations) so the
logic is visible; tests/ cross-check key functions against statsmodels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def sample_size_two_proportions(
    p_control: float, mde_rel: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True
) -> int:
    """Users per arm to detect a relative lift `mde_rel` on a conversion rate.

    Uses the standard formula with pooled variance under H0 and unpooled under H1:
        n = [z_a * sqrt(2 p̄ q̄) + z_b * sqrt(p1 q1 + p2 q2)]² / (p2 - p1)²
    """
    p1, p2 = p_control, p_control * (1 + mde_rel)
    pbar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = stats.norm.ppf(power)
    num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num**2 / (p2 - p1) ** 2))


def power_two_proportions(
    n_per_arm: float, p_control: float, mde_rel: float, alpha: float = 0.05, two_sided: bool = True
) -> float:
    p1, p2 = p_control, p_control * (1 + mde_rel)
    pbar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n_per_arm)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    return float(stats.norm.cdf((abs(p2 - p1) - z_a * se0) / se1))


def mde_for_sample_size(
    n_per_arm: int, p_control: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest relative lift detectable with the given users per arm (bisection)."""
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_two_proportions(n_per_arm, p_control, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Fixed-horizon tests
# ---------------------------------------------------------------------------


@dataclass
class Result:
    control: float
    treatment: float
    diff: float
    diff_ci: tuple[float, float]
    rel_lift: float
    rel_ci: tuple[float, float]
    p_value: float
    n_control: int
    n_treatment: int

    def as_row(self, name: str = "") -> dict:
        return {
            "metric": name,
            "control": self.control,
            "treatment": self.treatment,
            "abs_diff": self.diff,
            "rel_lift": self.rel_lift,
            "rel_ci_low": self.rel_ci[0],
            "rel_ci_high": self.rel_ci[1],
            "p_value": self.p_value,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
        }


def _rel_ci(m_c, m_t, var_mc, var_mt, z):
    """Delta-method CI for m_t / m_c - 1 (independent arms)."""
    rel = m_t / m_c - 1
    se = np.sqrt(var_mt / m_c**2 + m_t**2 * var_mc / m_c**4)
    return rel, (rel - z * se, rel + z * se)


def mean_test(y_c, y_t, alpha: float = 0.05) -> Result:
    """Welch z-test on means (large-sample). Works for 0/1 conversion too."""
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    m_c, m_t = y_c.mean(), y_t.mean()
    v_c, v_t = y_c.var(ddof=1) / y_c.size, y_t.var(ddof=1) / y_t.size
    z = stats.norm.ppf(1 - alpha / 2)
    d, se = m_t - m_c, np.sqrt(v_c + v_t)
    p = 2 * stats.norm.sf(abs(d / se))
    rel, rel_ci = _rel_ci(m_c, m_t, v_c, v_t, z)
    return Result(m_c, m_t, d, (d - z * se, d + z * se), rel, rel_ci, float(p), y_c.size, y_t.size)


def ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> Result:
    """Ratio-of-sums metric (e.g. orders per user-day, revenue per order) with
    delta-method variance, clustering at the user level.

    Var(R) ≈ (1 / (n μ_d²)) · (σ_n² − 2 R σ_nd + R² σ_d²)
    """
    z = stats.norm.ppf(1 - alpha / 2)

    def _r(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        n = num.size
        r = num.sum() / den.sum()
        mu_d = den.mean()
        cov = np.cov(num, den, ddof=1)
        var = (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1]) / (n * mu_d**2)
        return r, var, n

    r_c, v_c, n_c = _r(num_c, den_c)
    r_t, v_t, n_t = _r(num_t, den_t)
    d, se = r_t - r_c, np.sqrt(v_c + v_t)
    p = 2 * stats.norm.sf(abs(d / se))
    rel, rel_ci = _rel_ci(r_c, r_t, v_c, v_t, z)
    return Result(r_c, r_t, d, (d - z * se, d + z * se), rel, rel_ci, float(p), n_c, n_t)


def cuped(y, x, treat_mask=None):
    """CUPED: y_adj = y − θ (x − x̄), θ = cov(y, x) / var(x), pooled over both arms.

    x must be a pre-experiment covariate (unaffected by treatment), so θ can be
    estimated on pooled data without biasing the treatment effect.
    Returns (y_adj, theta, variance_reduction).
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    reduction = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    return y_adj, float(theta), float(reduction)


def srm_test(n_control: int, n_treatment: int, expected_share: float = 0.5) -> float:
    """Sample-ratio-mismatch chi-square p-value. p < 0.001 → stop and debug assignment."""
    n = n_control + n_treatment
    exp = [n * (1 - expected_share), n * expected_share]
    return float(stats.chisquare([n_control, n_treatment], exp).pvalue)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def interaction_z(rel_a, ci_a, rel_b, ci_b, alpha: float = 0.05) -> float:
    """Test whether two independent segment lifts differ (z-test on the difference)."""
    z = stats.norm.ppf(1 - alpha / 2)
    se_a, se_b = (ci_a[1] - ci_a[0]) / (2 * z), (ci_b[1] - ci_b[0]) / (2 * z)
    return float(2 * stats.norm.sf(abs(rel_a - rel_b) / np.hypot(se_a, se_b)))


# ---------------------------------------------------------------------------
# Sequential testing (the fix for peeking)
# ---------------------------------------------------------------------------


def obrien_fleming_bounds(n_looks: int, alpha: float = 0.05, n_sim: int = 200_000, seed: int = 0) -> np.ndarray:
    """O'Brien-Fleming boundaries for `n_looks` equally spaced looks.

    Boundary at look k is c · sqrt(K / k). The constant c is calibrated by
    Monte Carlo on the joint distribution of the cumulative z-statistics so that
    P(any |Z_k| > bound_k | H0) = alpha.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_looks + 1)
    incr = rng.standard_normal((n_sim, n_looks))
    z = np.cumsum(incr, axis=1) / np.sqrt(k)
    shape = np.sqrt(n_looks / k)
    max_stat = np.max(np.abs(z) / shape, axis=1)
    c = np.quantile(max_stat, 1 - alpha)
    return c * shape


def msprt_always_valid_p(diffs, variances, tau: float) -> np.ndarray:
    """Always-valid p-values from a mixture SPRT with a N(0, tau²) prior on the effect
    (Johari et al., 2017; normal approximation).

    At each look, with estimate d and its sampling variance v:
        Λ = sqrt(v / (v + τ²)) · exp(τ² d² / (2 v (v + τ²)))
    The always-valid p-value is the running min of 1/Λ. You may check it after
    every single user and stop the first time it is below alpha.
    """
    d, v = np.asarray(diffs, float), np.asarray(variances, float)
    lam = np.sqrt(v / (v + tau**2)) * np.exp(tau**2 * d**2 / (2 * v * (v + tau**2)))
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))
