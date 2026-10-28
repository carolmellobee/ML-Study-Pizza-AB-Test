"""Synthetic data generator for the pizza-app menu redesign experiment.

Produces two tables:

users  - one row per enrolled user: id, enrollment day, platform, region,
         assigned variant, pre-period order count (for CUPED) and median
         menu load time (guardrail).
orders - one row per order: user id, calendar day, order value.

Run as a module to write CSVs to ./data:

    python -m pizza_ab.simulate
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENROLL_DAYS, PLATFORMS, REGIONS, SALT, TEST_DAYS, TRUTH, GroundTruth

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def assign_variant(user_ids, salt: str = SALT) -> np.ndarray:
    """Deterministic 50/50 assignment by hashing salt + user id (how real platforms do it)."""
    out = np.empty(len(user_ids), dtype=object)
    for i, uid in enumerate(user_ids):
        h = hashlib.md5(f"{salt}:{uid}".encode()).hexdigest()
        out[i] = "treatment" if int(h[:8], 16) % 2 else "control"
    return out


def simulate(
    seed: int = 22,
    truth: GroundTruth = TRUTH,
    treatment_active: bool = True,
    enroll_days: int = ENROLL_DAYS,
    test_days: int = TEST_DAYS,
    id_prefix: str = "u",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # --- enrollment -------------------------------------------------------
    daily_n = rng.poisson(truth.users_per_day, size=enroll_days)
    enroll_day = np.repeat(np.arange(enroll_days), daily_n)
    n = enroll_day.size
    user_id = np.array([f"{id_prefix}{i:07d}" for i in range(n)])

    platform = rng.choice(PLATFORMS, size=n, p=[truth.platform_share[p] for p in PLATFORMS])
    region = rng.choice(REGIONS, size=n, p=truth.region_share)
    variant = assign_variant(user_id)
    is_t = (variant == "treatment") & treatment_active

    # --- user propensity --------------------------------------------------
    base = np.where(platform == "iOS", truth.base_daily_p["iOS"], truth.base_daily_p["Android"])
    k = truth.propensity_shape
    p_user = np.minimum(base * rng.gamma(k, 1 / k, size=n), 0.5)

    # Pre-period (90 days before the test): same propensity, no treatment.
    pre_orders = rng.binomial(90, p_user)

    # --- daily order draws over the test window ---------------------------
    days = np.arange(test_days)
    s = days[None, :] - enroll_day[:, None]            # days since first exposure
    active = s >= 0
    dow = np.asarray(truth.dow_multiplier)[days % 7][None, :]

    long_run = np.where(platform == "iOS", truth.long_run_lift["iOS"], truth.long_run_lift["Android"])
    novelty = truth.novelty_lift * np.exp(-np.clip(s, 0, None) / truth.novelty_tau_days)
    mult = np.where(is_t[:, None], (1 + long_run[:, None]) * (1 + novelty), 1.0)

    p = np.clip(p_user[:, None] * dow * mult, 0, 1)
    ordered = active & (rng.random(p.shape) < p)
    u_idx, day_idx = np.nonzero(ordered)

    mu = np.log(truth.aov_mean) - truth.aov_sigma**2 / 2
    value = np.round(rng.lognormal(mu, truth.aov_sigma, size=u_idx.size), 2)

    # --- guardrail: median menu load time ---------------------------------
    med = np.where(platform == "iOS", truth.load_median_s["iOS"], truth.load_median_s["Android"])
    added = np.where(platform == "iOS", truth.load_added_s["iOS"], truth.load_added_s["Android"])
    load_s = med * rng.lognormal(0, truth.load_sigma, size=n) + np.where(is_t, added, 0.0)

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "enroll_day": enroll_day,
            "platform": platform,
            "region": region,
            "variant": variant,
            "pre_orders_90d": pre_orders,
            "menu_load_s": np.round(load_s, 3),
        }
    )
    if not treatment_active:  # historical data has no experiment assignment
        users = users.drop(columns="variant")
    orders = pd.DataFrame({"user_id": user_id[u_idx], "day": day_idx, "order_value": value})
    return users, orders


def load_or_simulate(kind: str = "experiment") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs from ./data, generating them first if missing.

    kind="experiment": the A/B test.  kind="historical": a pre-test month with no
    treatment, used only for the power analysis.
    """
    u_path, o_path = DATA_DIR / f"{kind}_users.csv", DATA_DIR / f"{kind}_orders.csv"
    if not (u_path.exists() and o_path.exists()):
        write_all()
    return pd.read_csv(u_path), pd.read_csv(o_path)


def write_all() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for kind, kwargs in {
        "experiment": dict(seed=22, treatment_active=True, id_prefix="u"),
        "historical": dict(seed=2025, treatment_active=False, id_prefix="h"),
    }.items():
        users, orders = simulate(**kwargs)
        users.to_csv(DATA_DIR / f"{kind}_users.csv", index=False)
        orders.to_csv(DATA_DIR / f"{kind}_orders.csv", index=False)
        print(f"{kind}: {len(users):,} users, {len(orders):,} orders -> {DATA_DIR}")


if __name__ == "__main__":
    write_all()
