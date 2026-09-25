# StatsBomb 360 R1.2A final promotion decision

The analysis uses the existing 4,831 static event-linked goalkeeper distributions. The exact frozen xR join contains 4,802 actions across 115 matches. The augmentation uses the preregistered logit of clipped frozen rho and fixed option fields. All reported relationships are associational.

## Choice and robustness

| Competition | Direct share, 0 | Direct share, 1 | Direct share, 2+ | Monotone |
| --- | ---: | ---: | ---: | --- |
| WC2022 | 85.9% (n=256) | 43.0% (n=384) | 27.2% (n=1858) | YES |
| EURO2024 | 91.9% (n=223) | 57.1% (n=259) | 25.8% (n=1851) | YES |

Successful completed-pass sensitivity: **PASS** for monotone direct share overall and in both competitions.
Corridor sensitivity (1.0, 1.5 primary, 2.0m): **PASS** for monotone direct share. The primary corridor remains 1.5m.

## Frozen xR incremental information

The baseline and augmented predictions use identical action IDs in each scope. Five GroupKFold splits hold out whole matches. Confidence intervals use 5,000 paired match-cluster draws, preserving repeated match multiplicity and score ties.

| Scope | ΔAUC [95% CI] | ΔBrier [95% CI] | Δlog loss [95% CI] |
| --- | ---: | ---: | ---: |
| ALL | +0.0599 [+0.0418, +0.0778] | -0.0068 [-0.0095, -0.0041] | -0.0207 [-0.0281, -0.0133] |
| WC2022 | +0.0457 [+0.0219, +0.0688] | -0.0042 [-0.0076, -0.0008] | -0.0107 [-0.0195, -0.0019] |
| EURO2024 | +0.0636 [+0.0363, +0.0905] | -0.0088 [-0.0131, -0.0044] | -0.0301 [-0.0436, -0.0168] |

Direction improves for all three metrics in both tournaments: **YES**.

## Calibration and selection

The frozen xR option-count residuals and match-bootstrap intervals are in `xr_option_calibration_by_competition.csv`. These tables show whether the same residual pattern appears in both tournaments without claiming causality.

360 coverage is not random. The covered and uncovered samples differ notably in goal-kick and event-pressure shares; the exact figures are in `coverage_selection_final.csv`. No missingness correction was attempted.

## Decision

**MAIN PAPER: YES**. The required gate values are recorded in `promotion_decision.json`. The option-set result meets all stated promotion conditions.
**ABSTRACT: YES**. Candidate sentence: Visible goalkeeper option sets improved match-held-out retention prediction beyond frozen event-only xR in two open-data tournaments, while fewer clear short lanes aligned with more direct distribution.

Strongest academic statement: Static visible-option counts are associated with observed distribution choice, and the preregistered augmentation evaluates whether they add information beyond frozen event-only xR.

Strongest coach statement: When fewer short passing lanes were visibly clear, direct distributions were more common in the covered snapshots, subject to the competition, completed-pass, and corridor checks above.

Freeze frames cannot establish pre-event movement, off-camera options, or which choice was tactically best.


## Answers to the final promotion questions

1. WC2022 direct share at 0/1/2+ visible clear short options: 85.9% → 43.0% → 27.2%. The direction holds.
2. Euro2024 direct share at 0/1/2+: 91.9% → 57.1% → 25.8%. The direction holds.
3. Completed-pass-only direct share remains monotone: WC2022 72.9% → 22.7% → 12.2%; Euro2024 80.2% → 32.0% → 11.0%. This uses provider completion, not the R1 outcome.
4. Corridor sensitivity remains monotone: 1.0m: 90.3% → 45.6% → 28.1%; 1.5m: 88.7% → 48.7% → 26.5%; 2.0m: 86.1% → 48.3% → 24.7%. The 1.5m result stays primary.
5. The preregistered logit-rho augmentation improves pooled AUC, Brier and log loss: +0.0599 [+0.0418, +0.0778], -0.0068 [-0.0095, -0.0041], and -0.0207 [-0.0281, -0.0133] respectively.
6. The three improvements point in the improving direction in both tournaments: YES.
7. The pooled deltas and 95% intervals are shown in the table above; the probability-quality intervals exclude zero in the improving direction.
8. Frozen xR R1-minus-rho residuals at 0/1/2+ options are WC2022: -0.260 → -0.021 → +0.030; EURO2024: -0.214 → -0.077 → +0.053. The same negative-to-positive pattern appears in both tournaments; full match-cluster intervals are in the calibration table.
9. The 360 sample has 4,831 covered and 2,787 uncovered eligible distributions. Goal-kick and event-pressure shares differ sharply by coverage (see the coverage table); coverage is selected. The coverage table's retention column is provider completion, and its direct column retains the R1.2 provider pass-length proxy. It is not a matched causal comparison.
10. MAIN PAPER: YES under the exact stated decision rule, with selection disclosed.
11. ABSTRACT: YES under the exact stated decision rule. The candidate sentence above is a fact candidate, not a rewritten abstract.
12. Strongest academic sentence: On 4,802 matched event UUIDs, static visible-option fields improved match-held-out retention prediction beyond frozen xR in both tournaments, while observed direct share declined as clear short options increased.
13. Strongest coach sentence: In these covered snapshots, keepers went direct less often when more short passing lanes were visibly clear; this is an observed pattern, not a judgement of any individual choice.

Provider pass completion and accepted R1 differ on 181 of the 4,802 joined actions. The completed-pass sensitivity uses completion; xR validation and central-exit retention use R1.
