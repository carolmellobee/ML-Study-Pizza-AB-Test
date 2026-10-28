"""Experiment configuration and the simulator's hidden ground truth.

The analysis notebooks (01-05) never import `TRUTH`. They work only from the
generated data, the way a real analyst would. Notebook 06 compares the
estimates against `TRUTH` at the very end, to show the methods recover it.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Public experiment design (what the team pre-registered)
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "menu_redesign_v2"
SALT = "menu_redesign_v2_2026"  # hashing salt for user-level assignment

ENROLL_DAYS = 14   # users are enrolled (first exposure) on days 0..13
PRIMARY_READ_DAY = 21  # every enrolled user has a full 7-day window by day 20
TEST_DAYS = 28     # the test keeps running to day 27 for the novelty check
WINDOW_DAYS = 7    # primary metric: ordered within 7 days of first exposure

PLATFORMS = ["iOS", "Android"]
REGIONS = ["North", "South", "East", "West", "Central", "Coast"]

# Day 0 is a Monday. Pizza demand peaks Friday/Saturday.
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Hidden ground truth (used ONLY by the simulator and by notebook 06)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundTruth:
    users_per_day: int = 9_000
    platform_share: dict = field(default_factory=lambda: {"iOS": 0.55, "Android": 0.45})
    region_share: tuple = (0.22, 0.20, 0.18, 0.16, 0.14, 0.10)

    # Baseline daily order probability, by platform
    base_daily_p: dict = field(default_factory=lambda: {"iOS": 0.020, "Android": 0.016})
    # User heterogeneity: multiplier ~ Gamma(shape, 1/shape) so its mean is 1
    propensity_shape: float = 1.3
    dow_multiplier: tuple = (0.80, 0.80, 0.85, 0.95, 1.30, 1.45, 0.85)

    # Treatment effect on the daily order rate:
    #   rate_T / rate_C = (1 + long_run_lift[platform]) * (1 + novelty * exp(-s / tau))
    # where s = days since the user's first exposure.
    long_run_lift: dict = field(default_factory=lambda: {"iOS": 0.07, "Android": -0.06})
    novelty_lift: float = 0.12
    novelty_tau_days: float = 3.5

    # Average order value (no true treatment effect)
    aov_mean: float = 38.0
    aov_sigma: float = 0.35

    # Median menu-page load time per user, seconds (log-normal)
    load_median_s: dict = field(default_factory=lambda: {"iOS": 1.10, "Android": 1.60})
    load_sigma: float = 0.30
    load_added_s: dict = field(default_factory=lambda: {"iOS": 0.08, "Android": 0.90})


TRUTH = GroundTruth()
