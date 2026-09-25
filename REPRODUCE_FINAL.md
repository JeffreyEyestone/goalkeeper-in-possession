# Reproduce the final paper evidence

This repository supports two levels of reproduction:

1. **Fast verification** of the frozen publication evidence.
2. **Clean-source reproduction** of the final StatsBomb 360 extension from official Hudl/StatsBomb open-data JSON.

The final working environment was macOS arm64 / Python 3.12. The repository also provides a pip-compatible final requirements file.

## 1. Clone and create the environment

```bash
git clone https://github.com/JeffreyEyestone/goalkeeper-in-possession.git
cd goalkeeper-in-possession

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-final.txt
```

The exact recorded scientific environment is also preserved in `environment-wg13.yml`.

## 2. Fast verification

Run:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/verify_final_winner_lock.py
PYTHONPATH=src python scripts/verify_statsbomb360_r12b.py --archive
```

These commands verify the frozen event-only winner lock and the packaged R1.2B StatsBomb 360 attribution/provenance evidence.

## 3. Event-only final evidence

Core final publication tables are already frozen in:

- `results/final_winner_lock/`
- `results/winner_gate13/`

The canonical model/source map is documented in:

- `provenance/CANONICAL_MODEL_SOURCE_MAP.md`
- `results/canonical_source_discovery/model_source_map.csv`

Primary event-only verifier:

```bash
PYTHONPATH=src python scripts/verify_final_winner_lock.py
```

## 4. Full StatsBomb 360 clean-source reproduction

Official source:

https://github.com/hudl/open-data

The R1.2B provenance pipeline:

1. downloads every required official JSON file into a clean temporary raw directory from one pinned upstream commit;
2. requires HTTP 200, nonzero bytes, valid JSON and SHA256 for every file;
3. re-runs the frozen R1.2 option-table generator;
4. compares reproduced 4,831 distribution rows and 27,777 candidate-option rows against the frozen analysis tables;
5. evaluates nested match-grouped retention models and paired match-cluster bootstrap uncertainty.

Run in this order:

```bash
PYTHONPATH=src python scripts/download_statsbomb360_r12b_verified.py
PYTHONPATH=src python scripts/reproduce_statsbomb360_r12b.py
PYTHONPATH=src python scripts/run_statsbomb360_r12b_attribution.py
PYTHONPATH=src python scripts/verify_statsbomb360_r12b.py
```

The R1.2B scripts currently use a fixed clean temporary directory under `/private/tmp`, matching the validated macOS run. On another operating system, ensure `/private/tmp` exists and is writable before running the clean-source pipeline.

## 5. Core values used in the paper

- Visible options / direct choice: `results/statsbomb360_r12a/choice_stability_by_competition.csv`
- Final 360 incremental attribution: `results/statsbomb360_r12b/primary_attribution_validation.csv`
- Camera-control robustness: `results/statsbomb360_r12b/camera_control_validation.csv`
- Verified official-source manifest: `results/statsbomb360_r12b/source_manifest_verified.csv`
- Raw-to-derived reproduction audit: `results/statsbomb360_r12b/raw_to_derived_reproduction.csv`
- Event-only portability: `results/final_winner_lock/portability_final.csv`
- Independent failure asymmetry: `results/final_winner_lock/deep_independent_final.csv`
- Intended-target geometry: `results/final_winner_lock/intended_geometry_final.csv`

## 6. Paper artifacts

Submission-ready PDFs and abstract text are in `paper/`; source figures are in `figures/`.

## 7. Data attribution

Raw StatsBomb data are not bundled in this repository. See `DATA_SOURCES.md` for source and attribution requirements.
