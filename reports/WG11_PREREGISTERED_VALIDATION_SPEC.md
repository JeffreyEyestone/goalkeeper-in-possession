# Winner Gate 1.1 preregistered validation specification

This specification is frozen before primary result generation.

## Realized outcomes

Y10_NET_XG and Y30_NET_XG are signed observed StatsBomb shot-xG totals in the next 10 and 30 seconds of the same period after each goalkeeper distribution. YCYCLE_NET_XG is the signed observed shot-xG total through the end of the current possession and immediately following possession. YNEXTSHOT is the signed xG of the next shot within the next two possessions, or zero if none exists. No R1/R2/R3, rho, xT, V, or C enters these realized outcomes.

## Baselines

B0/B1 are keeper-level descriptive completion and R1 comparators only. B2 rho, B3 success-only, B4 training-only global failure cost, B5 training-only origin-band failure cost, and B6 canonical xT-GK use identical action IDs. TA-GEO is selected-sample sensitivity only.

## Eligibility and inference

Tournament keepers require >=2 matches and >=25 distributions; league keepers require >=200 distributions. Keeper bootstrap draws sample eligible keepers with replacement. Match-cluster bootstrap samples match IDs with replacement and recomputes pooled metrics. Deep cutoffs are frozen at 16.5m, 26.25m, and 35m. Independent cohort refits use frozen estimator settings and match cross-fitting. Shrinkage multipliers are 0.5, 1.0, and 2.0 and are passed into each refit.

## Interpretation

Criterion validity is associative. Positive results do not establish causal value or transfer success. The failure branch is interpreted as an explicit downside consequence, with origin value surrendered in both branches.
