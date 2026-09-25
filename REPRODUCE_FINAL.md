# Reproduce the final paper evidence

## 1. Environment
Use the frozen environment files in this repository. The final canonical event-model work was executed in the recorded scientific environment; StatsBomb 360 R1.2B records its own source/provenance lock.

## 2. Event-only final evidence
The repository contains frozen processed publication inputs and canonical OOF outputs. Run the existing event-model verification/test commands documented in the project scripts and `Makefile`.

## 3. StatsBomb 360 extension
Official source: https://github.com/hudl/open-data

The final R1.2B provenance pipeline:
1. downloads every required official JSON file into a clean temporary raw directory from the pinned upstream source;
2. requires HTTP success, nonzero bytes, valid JSON and SHA256;
3. re-runs the frozen R1.2 option-table generator;
4. compares reproduced 4,831 distribution rows and 27,777 candidate-option rows against the frozen analysis tables;
5. evaluates nested match-grouped retention models and paired match-cluster bootstrap uncertainty.

Relevant scripts include:
- `scripts/download_statsbomb360_r12b_verified.py`
- `scripts/reproduce_statsbomb360_r12b.py`
- `scripts/run_statsbomb360_r12b_attribution.py`
- `scripts/verify_statsbomb360_r12b.py`

## 4. Core final values used in the paper
- Visible options / direct choice: `results/statsbomb360_r12a/choice_stability_by_competition.csv`
- Incremental 360 information: `results/statsbomb360_r12b/primary_attribution_validation.csv`
- Camera-control robustness: `results/statsbomb360_r12b/camera_control_validation.csv`
- Event-only portability: `results/final_winner_lock/portability_final.csv`
- Independent failure asymmetry: `results/final_winner_lock/deep_independent_final.csv`
- Intended-target geometry: `results/final_winner_lock/intended_geometry_final.csv`

## 5. Paper artifacts
Submission-ready PDFs and abstract text are in `paper/`; source figures are in `figures/`.
