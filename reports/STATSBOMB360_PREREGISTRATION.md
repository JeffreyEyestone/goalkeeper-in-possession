# StatsBomb 360 Option-Set Pilot R1.2: preregistration

**CORE SCIENCE FROZEN = YES**

This is a separate, event-linked freeze-frame extension. It will not retrain or
alter xR-GK, V, C, xT-GK, WG1.3, the Final Winner Science Lock, or the R1.1
event-data coach product. It uses only the official Hudl StatsBomb open-data
repository and will not use SkillCorner or construct a universal goalkeeper
ranking.

## Frozen provenance

| Item | Value |
| --- | --- |
| Workspace commit at preregistration | `8799ab4cd236178a07372918e30ce450b54792d6` |
| Final Winner Lock record | `README_FINAL_WINNER_LOCK.md` |
| Final Winner Lock record SHA256 | `22862de2bd80a01d6025f9232a4d3aa0c671f65a0545dd4b55381e4b0f48de5f` |
| Final Sloan evidence record SHA256 | `e01ee5d725e151b3bb93d7c684a115990267aaa5c3532328f2b420459c7801bb` |
| R1.1 coach findings SHA256 | `647afb35252d3d30cbf9d929abee3b035339d02d6c0f37d9c0523b479cb0abd4` |
| Canonical frozen xR lock | `results/winner_gate13/CANONICAL_LOCK.json` |
| Canonical frozen xR lock SHA256 | `2f01615fa3cbb723d4363c88af142cac0e3bbb521c86e7fa80e36a8e4108a628` |

The canonical lock contains the frozen source and fitted-artifact SHA256
values. This pilot may attach frozen out-of-fold `rho` values by exact event
UUID only; it will never fit, select, or overwrite a core model.

## Fixed data and estimand

The source is `https://github.com/hudl/open-data`. The intended competitions
are FIFA World Cup 2022 (`competition_id=43`, `season_id=106`) and UEFA Euro
2024 (`competition_id=55`, `season_id=282`), subject to confirmation from the
downloaded registry and `match_available_360` availability in their official
match lists.

StatsBomb 360 frames are event-linked static snapshots, not continuous
tracking. Analyses will therefore describe **visible options**, **visible
lane-clear options**, and **visible central exits** only. They will not infer
movement, off-camera availability, tactical viability, or identify a central
player as a No. 6 without roster-position support.

Coordinates will be converted from the provider's 120 x 80 system to 105 x 68
metres. Candidate teammates exclude the event goalkeeper. Distance families
are short `<25m`, medium `25–40m`, and direct `>=40m`; width thirds are left
`<22.67m`, central `22.67–45.33m`, right `>45.33m`. The primary lane corridor
is fixed at 1.5m, with 1.0m and 2.0m sensitivities. All corridor tests require
the opponent projection to lie within the goalkeeper-to-candidate segment.

The choice and retention regressions are associational with match-clustered
standard errors. The frozen-xR comparison will use exactly the same joined
actions, five-fold match-grouped OOF evaluation, and a 5,000-draw paired
match-cluster bootstrap. It is an incremental-information evaluation model,
not a replacement xR model.

Main-paper eligibility is pre-specified as at least 500 usable distributions,
20 matches, both competitions represented, no match contributing over 10%, a
reproducible estimand, no single-competition-only direction, and explicit
coverage limitations. Incremental-xR eligibility additionally requires
consistent direction across AUC, Brier, and log loss plus at least one
probability-quality bootstrap interval excluding zero in the improving
direction. Otherwise findings are supplement/proof-of-concept only.
