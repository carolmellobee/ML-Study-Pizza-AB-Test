# Decision memo — Menu v2 experiment

**Decision: ship Menu v2 on iOS. Do not ship it on Android. Do not ship it to everyone.**

| | |
|---|---|
| Test | `menu_redesign_v2`: photo-card menu + "Popular near you" row |
| Design | User-level 50/50 split; 125,528 users enrolled over 14 days; primary read on day 21; test ran through day 27 |
| Pre-registered | Primary metric, MDE 5%, guardrails (with CI-bound margins), platform segment, weeks 2–4 decision metric, ship rule (notebook 01) |

## What we found

1. **The headline is real but misleading.** 7-day order conversion rose **+7.6%** (95% CI +4.4% to +10.9%,
   p ≈ 2·10⁻⁶). The split was clean (SRM p = 0.32) and pre-period covariates were balanced.
2. **Most of it is novelty.** On one consistent metric, orders per user-day, the lift is **+8.2% in users'
   first week** and **+2.3% in weeks 2–4** (95% CI −0.1% to +4.6%), which is not significant.
3. **Android is harmed.** Weeks 2–4 effect: **+8.3% on iOS** (CI +5.1% to +11.6%) and **−7.0% on Android**
   (CI −10.5% to −3.6%); interaction p ≈ 2·10⁻¹⁰. Novelty masked the Android loss in week 1
   (−1.2% on 7-day conversion, not significant).
4. **Likely cause: page weight.** On average the new menu loads 0.9 s slower on Android (+54%) and 0.08 s
   slower on iOS (+7%). At p75 the increases are +46% and +6%. The load-time guardrail fails overall and on
   Android, and passes on iOS. This is plausible but not proven causal: load time was changed by the treatment,
   not randomized. The Android re-test below is the causal check.
5. **Peeking would have shipped it on day 1**, reporting +24% with 14% of the planned sample.

## Expected impact (orders per user-day, weeks 2–4)

| Option | Change in total orders (95% CI) |
|---|---|
| Ship to everyone | +2.3% (−0.1% to +4.6%), and Android users lose ~7% |
| **Ship to iOS only** | **+5.1% (+3.1% to +7.0%)** (iOS is 61% of orders × +8.3%) |
| Ship to nobody | 0 |

These project from days 7–27 after exposure. Nothing here measures beyond week 4.

## Risks and next steps

* **The iOS estimate is probably optimistic**, because we are acting on the best-looking segment. Keep a 5% iOS
  holdout for 4 weeks after launch and report impact from it. That also tests whether the effect lasts past week 4.
* **Android:** ship smaller, lazy-loaded images; confirm load time is back within the guardrail; re-test on
  Android only, with the first 7 days excluded from the primary metric.
* **The guardrails were underpowered.** The AOV guardrail fails on Android by noise alone (−0.5%, lower bound
  −2.3%), and in re-runs it blocks a correct iOS ship 16% of the time. Size the guardrails in the next test.
* **Cost:** two menu versions must be maintained until Android is fixed or the redesign is dropped there.
* **Process:** for visual changes, default to a primary metric measured after a burn-in, and show always-valid
  p-values on the live dashboard.

*Evidence: notebooks 02–06. The data is simulated; see the README.*
