"""Why seed 22? Scan 40 seeds and show where the main experiment sits.

The repo's main experiment uses seed=22 because its headline numbers sit near the
median of this scan rather than at an extreme. Run:

    python scripts/seed_scan.py
"""

import pandas as pd

from pizza_ab.metrics import long_run_user, user_metrics
from pizza_ab.simulate import simulate
from pizza_ab.stats import mean_test, ratio_test

rows = []
for seed in range(1, 41):
    u, o = simulate(seed=seed)
    m = user_metrics(u, o)
    r = mean_test(m.loc[m.variant == "control", "converted"], m.loc[m.variant == "treatment", "converted"])
    row = {"seed": seed, "primary_lift": r.rel_lift}
    lr = long_run_user(u, o)
    for seg in ["All", "iOS", "Android"]:
        ll = lr if seg == "All" else lr[lr.platform == seg]
        c, t = ll[ll.variant == "control"], ll[ll.variant == "treatment"]
        row[f"long_run_{seg}"] = ratio_test(c.lr_orders, c.lr_days, t.lr_orders, t.lr_days).rel_lift
    rows.append(row)

df = pd.DataFrame(rows).set_index("seed")
summary = pd.DataFrame({"seed 22": df.loc[22], "median": df.median(),
                        "p10": df.quantile(0.1), "p90": df.quantile(0.9)})
print(summary.map(lambda x: f"{x:+.1%}"))
