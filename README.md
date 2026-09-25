# The Goalkeeper in Possession
## Visible Options, Retention Risk, and Asymmetric Turnover Cost

Public reproducibility repository for the MIT Sloan Sports Analytics Conference 2027 Soccer-track submission.

**Repository:** https://github.com/JeffreyEyestone/goalkeeper-in-possession

## What the study asks
Goalkeeper distribution is not only the pass observed after release. It is a decision made from an option set under asymmetric risk:

1. **What visible short/central options exist?**
2. **How likely is the selected distribution to retain possession?**
3. **How does that probability change across football environments?**
4. **What is the downside if the distribution fails?**

## Primary evidence

### Event-data study
- 45,974 goalkeeper distributions
- 722 matches
- Premier League 2015/16, FA WSL 2020/21, WC2018, WC2022, Euro/Copa America 2024

### StatsBomb 360 extension
- 4,831 event-linked goalkeeper distributions
- 115 matches from WC2022 and Euro2024
- 4,802 exact event UUID joins to frozen event-only xR

Main 360 findings:
- Direct play falls sharply as visible lane-clear short exits increase in both tournaments.
- Two fixed 360 option variables (clear-short count + central short exit) improve match-held-out retention prediction beyond locally recalibrated frozen xR, event pressure, goal-kick state, and competition.
- Pooled AUC: 0.6485 -> 0.6991; delta +0.0506, 95% paired match-cluster CI +0.0348 to +0.0660.

### Event-only portability and consequence
- True LOCO xR AUC: 0.645-0.710.
- Absolute probability scale shifts materially by competition.
- Independently refit opponent consequence C/V: 1.40-1.90 across five cohorts.
- Full lost branch (V+C)/V: 2.40-2.90.

## Quick verification

Recommended fast path:

```bash
git clone https://github.com/JeffreyEyestone/goalkeeper-in-possession.git
cd goalkeeper-in-possession

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-final.txt

PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/verify_final_winner_lock.py
PYTHONPATH=src python scripts/verify_statsbomb360_r12b.py --archive
```

For the full clean-source StatsBomb 360 provenance reproduction, see `REPRODUCE_FINAL.md`.

## Repository layout
- `src/` canonical xR/value implementation
- `scripts/` event-data and StatsBomb 360 generators / validators
- `tests/` reproducibility tests
- `data/manifests/` event-data match manifests
- `results/final_winner_lock/` final event-only evidence
- `results/statsbomb360/` frozen 360 option tables
- `results/statsbomb360_r12a/` tournament stability / sensitivity
- `results/statsbomb360_r12b/` final incremental-attribution and provenance lock
- `results/coach_product_r11/` descriptive coaching profiles
- `figures/` main paper and abstract figures
- `paper/` submission-ready paper / supplement / abstract
- `reports/` scientific audit reports
- `provenance/` model/source lineage

## Data sources
The underlying public data come from the official Hudl StatsBomb open-data repository. Raw provider JSON is not duplicated here; exact source URLs, pinned upstream commit, SHA256 values, and clean reproduction checks are in `results/statsbomb360_r12b/source_manifest_verified.csv` and the source download scripts.

Official source: https://github.com/hudl/open-data

StatsBomb's open-data terms ask published work to state StatsBomb as the data source and use the StatsBomb logo. See `DATA_SOURCES.md`.

## Reproducibility
See `REPRODUCE_FINAL.md`.

## Interpretation guardrails
- StatsBomb 360 is an event-linked freeze frame, not continuous tracking.
- Off-camera players are unobserved, not unavailable.
- Option/choice relationships are observational, not causal pressing effects.
- xR is an action-risk probability, not pure goalkeeper technique.
- Full xT-GK is used as a consequence/decision-value layer, not claimed as a superior universal goalkeeper ranking.
- Intended-target analyses are selected because failed actions expose targets less often.

## AI-assisted research disclosure
See `AI_USE_DISCLOSURE.md`.

## Citation
See `CITATION.cff`.

## License
Project software/code is released under the MIT License. Third-party data remain subject to their source-provider terms.
