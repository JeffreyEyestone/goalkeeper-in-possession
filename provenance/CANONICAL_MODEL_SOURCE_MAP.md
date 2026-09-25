# Canonical Model Source Map

This document is the human-readable companion to:

`results/canonical_source_discovery/model_source_map.csv`

It records the executable implementations used by the accepted event-data framework before the final publication evidence was frozen.

| Component | Accepted quantity | Canonical source | Trainer / orchestrator | Fitted artifact | Refit status |
|---|---|---|---|---|---|
| TF xR-GK | `rho_TF` | `src/gkpossession/xr.py` (`classifier`, `fit_calibration`, `calibrated`) | `scripts/fit_xr.py::main` | `data/processed/models/xr_TF.joblib` | REFIT_READY |
| Origin / success value | `origin_value_TF`, `success_payoff_TF` | `src/gkpossession/surfaces.py` (`fit_bundle`, `lookup`) | `scripts/fit_values.py::main` | `data/processed/models/surface.joblib` | REFIT_READY |
| TF failure consequence | `failure_cost_TF` | `src/gkpossession/failure_cost.py` (`ShrunkBranchMean`, `regressor`, `model_for`) | `scripts/fit_values.py::main` | `data/processed/models/branches_TF.joblib` | REFIT_READY |
| TA-GEO xR | `rho_TA` | `src/gkpossession/xr.py` + target-aware orchestration | `scripts/fit_xr.py::main`, `scripts/receiver_extension.py::main` | `data/processed/models/xr_TA.joblib` | REFIT_READY |
| TA failure consequence | `failure_cost_TA` | `src/gkpossession/failure_cost.py` | `scripts/fit_values.py::main` | `data/processed/models/branches_TA.joblib` | REFIT_READY |

## Canonical scoring

The accepted risk-aware value composition is implemented in:

`src/gkpossession/xtgk.py`

The reporting form is:

`xT-GK = rho * success_payoff - origin_value - (1-rho) * failure_cost`

The event-only publication evidence treats xR-GK as the retention-risk probability component and xT-GK as a consequence/decision-value layer; the final paper does not claim that full xT-GK is a superior universal goalkeeper ranking.

## Frozen artifact hashes from the source-discovery lock

- TF xR artifact SHA256: `5faef510caabe8e2764ffd243a31e37217e332b68db7887f77d233891d8661b7`
- Value surface SHA256: `8dc48ba2819d0cae2db16dbd56555f448c34abd52adc3c738ebbd18abcb04560`
- TF branch artifact SHA256: `f734a5fff5c4253a4cd5551227858c694fa92256fd82ef435d3f042dd4756ab5`
- TA xR artifact SHA256: `a77e11cff2cae4b2f65a0b19327b6d0c95267053f504d59ec8f201ac4d8924bb`
- TA branch artifact SHA256: `2b812b1383bdc4aed6555ba7a60efb9150632800f573240f3a256168ec5b2aa9`

Source-discovery commit recorded in the canonical map:

`4958c43c64f05602dc356ecc435238d0b1bf9fb9`

For exact feature lists, selection logic and notes, use the machine-readable CSV above.
