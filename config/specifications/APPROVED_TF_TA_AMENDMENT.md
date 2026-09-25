APPROVED: Option 1 — target-free primary model with target-aware secondary analysis.

Proceed from checkpoint 00fe615.

However, approval is subject to the following scientific amendments. These are now part of the governing specification.

1. PRIMARY AND SECONDARY ESTIMANDS

Implement and preserve TWO explicitly different estimands throughout the code and reports.

Internally label them:

TF = target-free, all-action model
TA = target-aware, intended-target model

Do not conflate them.

TF is the PRIMARY scientific model because it preserves the full distribution population and does not condition analysis on target availability.

TA is a SECONDARY/SENSITIVITY model using only actions for which a defensible intended target can be reconstructed.

Do not silently call the TF model “pass-specific intended-target difficulty.” Its interpretation is:

expected retention/value conditional on the pre-action state and observable chosen action family/context.

Its purpose is to answer whether a goalkeeper/team's distribution policy is valuable under the observed state/action family without using outcome-contaminated destination information.

The TA model may answer the finer intended-destination question, but only on the identifiable subset.

Do not decide final paper nomenclature yet. Keep TF and TA internal until results show what each can support.

2. DO NOT IMPUTE MISSING INTENDED TARGETS

Do not fill missing targets using realized endpoints.

Do not create model-based intended-target imputations for the primary analysis.

Do not assume missing-at-random.

The fact that target missingness is concentrated among unsuccessful actions is a substantive selection mechanism and must be documented.

Create:

reports/TARGET_MISSINGNESS_AND_SELECTION.md

Analyze target availability by:
- cohort
- success/failure
- restart/open play
- pressure
- pass height/technique/action family
- goalkeeper
- team
- match
- era
- any other pre-action characteristic available

Quantify how strongly target availability depends on outcome.

3. COMPARE TF AND TA WHERE BOTH CAN BE CALCULATED

On the overlap sample, run both models and compare:

- probability calibration
- discrimination
- keeper aggregates
- rank correlation
- xT-GK/value aggregates
- major headline relationships
- restart conclusions
- repeatability where sample permits

This comparison is important.

We need to know whether the all-action TF estimand preserves broadly similar keeper/value structure or whether intended destination materially changes conclusions.

Create:

reports/TARGET_FREE_VS_TARGET_AWARE.md

If the two approaches materially disagree, do not hide the disagreement. Explain what information intended target contributes.

4. RETENTION LABEL STUDY STILL COMES BEFORE MODEL FITTING

Approval of Option 1 does NOT pre-approve the legacy 10-second label.

Proceed with R1/R2/R3 as previously specified:

R1 = direct/controlled distribution resolution
R2 = 5-second team retention / shot
R3 = legacy 10-second team retention / shot

Add 15-second sensitivity if inexpensive.

Select the primary label based on conceptual congruence and measurement quality, not on which produces preferred historical results.

5. TARGET-FREE FEATURE DISCIPLINE

The TF primary model may use only information legitimately available at release / as part of the chosen action family.

Audit every feature.

Outcome-derived endpoint geometry is forbidden.

Receiver pressure is forbidden in the ex-ante primary model.

Any distance, angle, forward delta, lateral delta or destination-zone feature derived from realized end_location is forbidden from TF.

Action descriptors such as restart type, distribution technique, ball height/body part may be used only if they genuinely characterize the chosen action rather than its result; document the timing/semantics.

Create automated leakage tests.

6. DISTANCE / UNIT CORRECTION

The historical “meters” issue is now a mandatory correction.

Do NOT tune units or bin edges to recover the old 40–60 result.

First document exactly what StatsBomb's coordinate system represents in this feed.

Do not simply assume that 120x80 numerical coordinates are literal physical yards unless authoritative source semantics establish that.

For the primary paper, if physical meters are required and no match-specific physical pitch dimensions exist, use an explicitly declared standardized pitch mapping, preferably:

x: 0–120 -> 0–105 m
y: 0–80 -> 0–68 m

Compute physical-like Euclidean distance AFTER separately scaling x and y:

dx_m = dx_native * 105/120
dy_m = dy_native * 68/80
distance_m = sqrt(dx_m^2 + dy_m^2)

Call these “standardized pitch meters” or equivalent, not measured physical distance.

Also retain the native-coordinate calculation as a sensitivity analysis.

If authoritative StatsBomb documentation establishes a different convention, document it and explain the choice before proceeding.

Audit every existing distance-based figure/table/result.

7. THE HISTORICAL 40–60 M CLAIM IS NOW DISABLED

Treat the old claim that 40–60 m is the costliest restart as a failed-to-be-verified historical hypothesis until the corrected analysis is complete.

Do not preserve that band result because it was previously prominent.

Run:

- corrected fixed metric bands
- continuous/smoothed distance analysis
- uncertainty intervals
- cohort replication
- context-adjusted analysis
- TF/TA comparison where possible
- native-unit sensitivity

If WSL no longer shows the historical ordering, report that clearly.

The final paper claim must reflect corrected results.

Do not use “coach out the halfway ball” unless the corrected contextual evidence independently supports such a statement.

8. PRIMARY xR-GK INTERPRETATION

For TF:

xR-GK is NOT automatically “pass execution skill.”

It is an expected retention probability given state/action-family information.

Player execution is still:

xR-GK+ / Retention Over Expected
= observed retention – predicted retention

with appropriate aggregation and shrinkage.

Test whether xR-GK+ is actually repeatable before calling it a goalkeeper skill.

If it is not repeatable, report that result rather than forcing an execution metric.

9. xT-GK INTERPRETATION

The TF xT-GK primary should be interpreted as expected value of the observed distribution policy/action family in that context.

It must remain fully ex ante.

Neither the retained-payoff branch nor the failure-cost branch may use the realized outcome of the particular action being valued.

Target-aware TA may estimate a finer intended-pass value on the defensible-target subset.

Keep these concepts separate in code and reports.

10. EXPECTED FAILURE CONSEQUENCE

Continue with the revised C(a) approach.

C(a) must be expected failure consequence conditional on legitimate ex-ante/action-family information.

Do not use the action's realized failure endpoint when generating its expected decision value.

Realized failure location may be used as the TARGET for fitting/validating the failure-cost model on historical failures, but never as an INPUT when scoring that same action ex ante.

Add explicit tests proving this.

11. ADD TWO NEW STOP CONDITIONS

STOP A — TARGET-FREE INFORMATION FAILURE

If the TF model cannot produce useful probability calibration/discrimination or generates essentially no meaningful variation because target information was critical, do not continue pretending it solves the original problem.

Pause and report:

- quantitative performance
- information apparently missing
- whether the research question needs narrowing
- whether external target data become necessary

STOP B — TF/TA STRUCTURAL DISAGREEMENT

If TF and TA give fundamentally different keeper/value structures on the common sample, pause before manuscript conclusions.

Provide the exact differences and recommended interpretation.

12. COMPLETE ALL REMAINING PHASES AFTER THESE CHANGES

After implementing the above, continue through:

- retention-label decision
- xR-GK fitting/calibration
- value surface
- expected failure consequence
- xR-GK+
- xT-GK TF
- xT-GK TA sensitivity
- baselines
- uncertainty
- repeatability
- construct validity
- deep-consequence replication
- corrected restart analysis
- robustness matrix
- H1–H7 claim tests
- manuscript claim audit
- canonical results manifest
- final figures
- final scientific summary

Do not stop again merely because an old headline result changes.

Only stop for a genuine decision gate under the governing specification or one of the two new stop conditions above.

13. SPECIFIC QUESTION TO ANSWER IN FINAL SCIENTIFIC SUMMARY

The final analysis must clearly answer:

“Does open event data without reliable intended target information support a useful all-action valuation of goalkeeper distribution policy, and what additional inference becomes possible when intended target is identifiable?”

This distinction may itself become an academically important result.

Proceed now.