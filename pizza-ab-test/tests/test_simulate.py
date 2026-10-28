import numpy as np
import pandas as pd

from pizza_ab.config import WINDOW_DAYS
from pizza_ab.metrics import long_run_user, user_metrics
from pizza_ab.simulate import assign_variant, simulate
from pizza_ab.stats import srm_test


def test_simulation_is_deterministic():
    u1, o1 = simulate(seed=5)
    u2, o2 = simulate(seed=5)
    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(o1, o2)


def test_assignment_is_deterministic_and_balanced():
    ids = [f"u{i:07d}" for i in range(100_000)]
    a = assign_variant(ids)
    assert (a == assign_variant(ids)).all()
    assert srm_test((a == "control").sum(), (a == "treatment").sum()) > 0.001


def test_historical_has_no_assignment():
    u, _ = simulate(seed=5, treatment_active=False)
    assert "variant" not in u.columns


def test_user_metrics_window():
    u, o = simulate(seed=5)
    m = user_metrics(u, o)
    o = o.merge(u[["user_id", "enroll_day"]], on="user_id")
    in_window = o[(o.day >= o.enroll_day) & (o.day < o.enroll_day + WINDOW_DAYS)]
    assert m["orders"].sum() == len(in_window)
    assert set(m.loc[m.converted == 1, "user_id"]) == set(in_window["user_id"])


def test_long_run_excludes_first_week():
    u, o = simulate(seed=5)
    lr = long_run_user(u, o)
    assert (lr["lr_days"] > 0).all()
    assert lr["lr_days"].max() == 27 - 0 - 7 + 1


def test_orders_only_after_enrollment():
    u, o = simulate(seed=5)
    merged = o.merge(u[["user_id", "enroll_day"]], on="user_id")
    assert (merged["day"] >= merged["enroll_day"]).all()
    assert np.isfinite(o["order_value"]).all()


def test_novelty_fit_detects_decay():
    from pizza_ab.novelty import fit_novelty

    u, o = simulate(seed=5)
    fit = fit_novelty(u, o, n_boot=20)
    assert fit.loc["N", "estimate"] > 0
    assert fit.loc["N", "ci_low"] < fit.loc["N", "estimate"] < fit.loc["N", "ci_high"]
