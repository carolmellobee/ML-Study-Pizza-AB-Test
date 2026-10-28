"""Turn raw users/orders tables into analysis-ready metric tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEST_DAYS, WINDOW_DAYS


def _orders_with_exposure(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.merge(users[["user_id", "enroll_day"]], on="user_id", how="inner")
    o["s"] = o["day"] - o["enroll_day"]  # days since first exposure
    return o


def user_metrics(users: pd.DataFrame, orders: pd.DataFrame, window: int = WINDOW_DAYS) -> pd.DataFrame:
    """Primary/secondary metrics over each user's first `window` days after exposure.

    Columns added: converted (0/1), orders, revenue, aov_num (= revenue) for ratio AOV.
    Every user gets the same-length window, so there is no follow-up bias between
    early and late enrollees.
    """
    o = _orders_with_exposure(users, orders)
    o = o[(o["s"] >= 0) & (o["s"] < window)]
    agg = o.groupby("user_id").agg(orders=("day", "size"), revenue=("order_value", "sum"))
    out = users.merge(agg, on="user_id", how="left").fillna({"orders": 0, "revenue": 0.0})
    out["orders"] = out["orders"].astype(int)
    out["converted"] = (out["orders"] > 0).astype(int)
    return out


def exposure_curve(
    users: pd.DataFrame,
    orders: pd.DataFrame,
    by: str | None = None,
    last_day: int = TEST_DAYS - 1,
) -> pd.DataFrame:
    """Orders and user-days at risk by arm and days-since-exposure (s).

    A user enrolled on day e contributes one user-day at risk for every
    s = 0 .. last_day - e. Returns one row per (variant, [by], s).
    """
    keys = ["variant"] + ([by] if by else [])
    o = _orders_with_exposure(users, orders)
    o = o[(o["s"] >= 0) & (o["day"] <= last_day)]
    n_orders = o.merge(users[["user_id"] + keys], on="user_id").groupby(keys + ["s"]).size()

    # user-days at risk: count users per (keys, enroll_day), expand over s
    cnt = users.groupby(keys + ["enroll_day"]).size().rename("n").reset_index()
    rows = []
    for _, r in cnt.iterrows():
        for s in range(0, last_day - int(r["enroll_day"]) + 1):
            rows.append({**{k: r[k] for k in keys}, "s": s, "user_days": int(r["n"])})
    at_risk = pd.DataFrame(rows).groupby(keys + ["s"])["user_days"].sum()

    out = pd.concat([at_risk, n_orders.rename("orders")], axis=1).fillna(0).reset_index()
    out["orders"] = out["orders"].astype(int)
    out["rate"] = out["orders"] / out["user_days"]
    return out


def long_run_user(
    users: pd.DataFrame, orders: pd.DataFrame, s_min: int = 7, last_day: int = TEST_DAYS - 1
) -> pd.DataFrame:
    """Per-user orders and exposure-days counting only days s >= s_min after exposure.

    This drops each user's first week, where the novelty effect lives.
    Analysed as a ratio metric (sum orders / sum days) with the delta method.
    """
    o = _orders_with_exposure(users, orders)
    o = o[(o["s"] >= s_min) & (o["day"] <= last_day)]
    n_orders = o.groupby("user_id").size().rename("lr_orders")
    out = users.merge(n_orders, on="user_id", how="left").fillna({"lr_orders": 0})
    out["lr_orders"] = out["lr_orders"].astype(int)
    out["lr_days"] = np.clip(last_day - out["enroll_day"] - s_min + 1, 0, None)
    return out[out["lr_days"] > 0]


def dashboard_snapshot(users: pd.DataFrame, orders: pd.DataFrame, as_of_day: int) -> pd.DataFrame:
    """What a naive live dashboard shows at the end of `as_of_day`:
    users enrolled so far and whether they have ordered *so far* (unequal follow-up)."""
    u = users[users["enroll_day"] <= as_of_day]
    o = _orders_with_exposure(u, orders)
    o = o[(o["s"] >= 0) & (o["day"] <= as_of_day)]
    buyers = set(o["user_id"])
    u = u.assign(converted=u["user_id"].isin(buyers).astype(int))
    return u
