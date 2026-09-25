# StatsBomb 360 R1.2B final attribution and provenance lock

This analysis uses the frozen 4,831 event-linked distributions, 4,802 exact frozen-xR UUID joins and 115 matches. It isolates the contribution of two fixed 360 option fields above local calibration, event pressure, goal kick and competition. All results are observational associations, evaluated out of match on identical actions and folds.

## 1–2. Local calibration and non-360 terms

M0 is the unchanged frozen probability. M1 fits local calibration of `logit_rho` (plus competition in the pooled fit). M2 also re-estimates event `under_pressure` and `goal_kick`. These are non-360 controls. Changes are shown as upper model minus lower model:

| Step (pooled) | ΔAUC | ΔBrier | Δlog loss |
| --- | ---: | ---: | ---: |
| M1 − M0, local calibration | -0.0022 | +0.0003 | -0.0317 |
| M2 − M1, event-state controls | +0.0088 | -0.0004 | -0.0017 |
| M3 − M2, core 360 option fields | +0.0506 | -0.0065 | -0.0167 |

The attribution is **M3 versus M2**, because those models differ only in `lane_clear_short_options` and `central_short_exit`. M4 additionally contains visible teammate count and is secondary because that count correlates with camera coverage.

## 3–6. Core 360 incremental information

| Scope | M3−M2 ΔAUC [95% CI] | ΔBrier [95% CI] | Δlog loss [95% CI] |
| --- | ---: | ---: | ---: |
| ALL | +0.0506 [+0.0348, +0.0660] | -0.0065 [-0.0087, -0.0042] | -0.0167 [-0.0221, -0.0111] |
| WC2022 | +0.0548 [+0.0335, +0.0763] | -0.0055 [-0.0086, -0.0026] | -0.0137 [-0.0215, -0.0064] |
| EURO2024 | +0.0601 [+0.0353, +0.0829] | -0.0081 [-0.0118, -0.0044] | -0.0204 [-0.0289, -0.0112] |

The M3−M2 improvement points in the improving direction for all three metrics in WC2022 and Euro2024: **YES**. Five-thousand paired match-cluster draws retain repeated match multiplicity and tied-score AUC credit. Tournament-level significance is not required by the decision rule.

## 7. Camera-coverage robustness

Visible-area polygons were converted from 120×80 provider coordinates to 105×68m and measured by the shoelace formula. Pooled Pearson correlation between visible-area fraction and raw visible teammate count is +0.694. The complete area correlations are in `visible_area_audit.csv`. M2C adds visible-area fraction to M2; M3C adds only the two core 360 fields to M2C.

| Scope | M3C−M2C ΔAUC [95% CI] | ΔBrier [95% CI] | Δlog loss [95% CI] |
| --- | ---: | ---: | ---: |
| ALL | +0.0528 [+0.0367, +0.0687] | -0.0065 [-0.0086, -0.0043] | -0.0166 [-0.0218, -0.0113] |
| WC2022 | +0.0579 [+0.0360, +0.0798] | -0.0057 [-0.0087, -0.0027] | -0.0146 [-0.0225, -0.0071] |
| EURO2024 | +0.0475 [+0.0257, +0.0692] | -0.0074 [-0.0107, -0.0040] | -0.0183 [-0.0261, -0.0100] |

Camera-control robustness improves all three pooled metrics: **YES**. This is a robustness analysis added after the original preregistration.

## 8. Fixed component ablation

| Added to M2, pooled | ΔAUC | ΔBrier | Δlog loss |
| --- | ---: | ---: | ---: |
| clear-short count | +0.0481 | -0.0059 | -0.0144 |
| central exit | +0.0414 | -0.0048 | -0.0133 |
| both primary fields | +0.0506 | -0.0065 | -0.0167 |

Clear-short count is the larger single pooled component across these metrics; central exit also adds information. Correlated features prevent causal allocation of the combined gain.

## 9–12. Scientific hierarchy and abstract wording

- **Short/direct visible-option relationship: MAIN.** The accepted R1.2A cross-tournament, completed-pass and corridor findings are unchanged. Fresh raw data reproduce the frozen option tables field for field.
- **Frozen-xR option-count calibration: SUPPORTING DIAGNOSTIC.** The existing R1.2A R1-minus-rho pattern remains unchanged (WC2022: −0.260, −0.021, +0.030; Euro2024: −0.214, −0.077, +0.053 for 0/1/2+ clear-short options). This is not causal evidence.
- **360 information above non-360 control: MAIN.** M3 versus M2 passes the directional and pooled probability-quality interval rule. The pooled camera-control comparison also improves.
- **Abstract 360 claim approved: YES.** Candidate fact sentence: In two open-data tournaments, visible clear-short and central-exit fields improved match-held-out retention prediction beyond locally recalibrated frozen xR and event-state controls.

Exact main-paper claim: On 4,802 exactly joined goalkeeper distributions, two fixed visible-option fields improved match-held-out R1 prediction beyond locally recalibrated frozen xR, event pressure, goal kick and competition; the direction held in WC2022 and Euro2024 and after visible-area control. The descriptive choice relationship is separately supported, with selected 360 coverage disclosed.

## 13–14. Raw-source provenance

All 348 official JSON downloads returned HTTP 200, nonzero bytes, valid JSON and SHA256 from pinned Hudl StatsBomb open-data commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. The old R1.2 manifest had 13 zero-byte records; the verified manifest has zero. Raw provider files are not packaged.

The unmodified frozen R1.2 option-table generator was executed through its two table writes against the fresh official files in temporary storage. All 35 audited key and scientific-field checks match; the fresh outputs contain 4,831 distributions and 27,777 candidates. Parquet byte identity was not required.

## OOF model metrics

| Scope | Model | AUC | Brier | log loss |
| --- | --- | ---: | ---: | ---: |
| ALL | M0 | 0.6420 | 0.1661 | 0.5380 |
| ALL | M1 | 0.6397 | 0.1664 | 0.5063 |
| ALL | M2 | 0.6485 | 0.1661 | 0.5046 |
| ALL | M3 | 0.6991 | 0.1596 | 0.4879 |
| ALL | M4 | 0.7019 | 0.1594 | 0.4870 |
| ALL | M2C | 0.6453 | 0.1662 | 0.5051 |
| ALL | M3C | 0.6981 | 0.1597 | 0.4884 |
| WC2022 | M0 | 0.6451 | 0.1653 | 0.5122 |
| WC2022 | M1 | 0.6358 | 0.1660 | 0.5020 |
| WC2022 | M2 | 0.6371 | 0.1665 | 0.5030 |
| WC2022 | M3 | 0.6919 | 0.1610 | 0.4893 |
| WC2022 | M4 | 0.6908 | 0.1611 | 0.4898 |
| WC2022 | M2C | 0.6362 | 0.1665 | 0.5032 |
| WC2022 | M3C | 0.6941 | 0.1609 | 0.4886 |
| EURO2024 | M0 | 0.6436 | 0.1670 | 0.5657 |
| EURO2024 | M1 | 0.6267 | 0.1682 | 0.5140 |
| EURO2024 | M2 | 0.6339 | 0.1673 | 0.5096 |
| EURO2024 | M3 | 0.6940 | 0.1592 | 0.4892 |
| EURO2024 | M4 | 0.7072 | 0.1582 | 0.4854 |
| EURO2024 | M2C | 0.6493 | 0.1664 | 0.5070 |
| EURO2024 | M3C | 0.6968 | 0.1590 | 0.4887 |

360 coverage is selected: the R1.2A covered and uncovered samples differ substantially in goal-kick and event-pressure shares. Freeze frames cannot establish pre-event movement, off-camera options or a best tactical choice.

Data source: [Hudl StatsBomb open-data](https://github.com/hudl/open-data).
