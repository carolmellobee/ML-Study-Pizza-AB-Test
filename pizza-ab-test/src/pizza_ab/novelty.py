"""Novelty-effect diagnostics: lift by days since first exposure and a decay-model fit.

The per-day lifts share users (a user appears at every s they are observed), so the
28 points are NOT independent. `fit_novelty` reports a user-level bootstrap CI; the
curve_fit standard errors (which assume independence) are kept only for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .config import TEST_DAYS


def novelty_model(s, L, N, tau):
    """log rate ratio at s days since exposure: log(1 + L) + log(1 + N·exp(-s/τ))."""
    return np.log1p(L) + np.log1p(N * np.exp(-np.asarray(s, float) / tau))


def _user_matrices(users: pd.DataFrame, orders: pd.DataFrame, last_day: int = TEST_DAYS - 1):
    """Per-user matrices over s = 0..last_day: orders[i, s] and at_risk[i, s] (0/1)."""
    idx = pd.Series(np.arange(len(users)), index=users["user_id"])
    e = users["enroll_day"].to_numpy()
    o = orders.merge(users[["user_id", "enroll_day"]], on="user_id")
    o = o[(o["day"] >= o["enroll_day"]) & (o["day"] <= last_day)]
    n_s = last_day + 1
    order_m = np.zeros((len(users), n_s))
    np.add.at(order_m, (idx[o["user_id"]].to_numpy(), (o["day"] - o["enroll_day"]).to_numpy()), 1)
    at_risk = (np.arange(n_s)[None, :] <= (last_day - e)[:, None]).astype(float)
    is_t = (users["variant"] == "treatment").to_numpy()
    return order_m, at_risk, is_t


def _lift_from_weights(order_m, at_risk, is_t, w):
    ot, oc = w[is_t] @ order_m[is_t], w[~is_t] @ order_m[~is_t]
    dt, dc = w[is_t] @ at_risk[is_t], w[~is_t] @ at_risk[~is_t]
    return (ot / dt) / (oc / dc) - 1, ot, oc


def lift_by_exposure_day(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Treatment/control daily order-rate ratio for each day s since first exposure.

    The per-point CI uses the Poisson log rate-ratio SE, sqrt(1/O_T + 1/O_C). Within a
    single s each user contributes at most one user-day, so this is fine point by point.
    """
    order_m, at_risk, is_t = _user_matrices(users, orders)
    lift, ot, oc = _lift_from_weights(order_m, at_risk, is_t, np.ones(len(users)))
    out = pd.DataFrame({"lift": lift, "se_log": np.sqrt(1 / ot + 1 / oc)}, index=pd.Index(np.arange(lift.size), name="s"))
    out["lo"] = np.expm1(np.log1p(out["lift"]) - 1.96 * out["se_log"])
    out["hi"] = np.expm1(np.log1p(out["lift"]) + 1.96 * out["se_log"])
    return out


def _fit(s, lift, se_log):
    return curve_fit(
        novelty_model, s, np.log1p(lift), p0=[0.02, 0.1, 3], sigma=se_log,
        absolute_sigma=True, bounds=([-0.5, -0.5, 0.3], [0.5, 1, 30]),
    )


def fit_novelty(users: pd.DataFrame, orders: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Fit `novelty_model` and return point estimates with user-level bootstrap CIs.

    Bootstrap: Poisson(1) weights per user (resampling users, not days), refit each time,
    take the 2.5th / 97.5th percentiles. Also returns the naive curve_fit CIs, which treat
    the per-day points as independent and are too narrow.
    """
    order_m, at_risk, is_t = _user_matrices(users, orders)
    lift, ot, oc = _lift_from_weights(order_m, at_risk, is_t, np.ones(len(users)))
    s = np.arange(lift.size)
    se_log = np.sqrt(1 / ot + 1 / oc)
    params, cov = _fit(s, lift, se_log)
    naive_se = np.sqrt(np.diag(cov))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, len(users)).astype(float)
        bl, bot, boc = _lift_from_weights(order_m, at_risk, is_t, w)
        try:
            boots.append(_fit(s, bl, np.sqrt(1 / bot + 1 / boc))[0])
        except RuntimeError:  # rare non-convergence; skip the draw
            continue
    boots = np.array(boots)
    return pd.DataFrame(
        {
            "estimate": params,
            "ci_low": np.percentile(boots, 2.5, axis=0),
            "ci_high": np.percentile(boots, 97.5, axis=0),
            "naive_ci_low": params - 1.96 * naive_se,
            "naive_ci_high": params + 1.96 * naive_se,
        },
        index=["L", "N", "tau"],
    )
