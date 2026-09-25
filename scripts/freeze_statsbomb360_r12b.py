"""Write the immutable R1.2B input and model-specification lock before analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/statsbomb360_r12b"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        ROOT / "results/statsbomb360/distribution_option_summary.parquet",
        ROOT / "results/statsbomb360/option_candidates.parquet",
        ROOT / "results/winner_gate13/oof_action_values.parquet",
        ROOT / "results/winner_gate13/CANONICAL_LOCK.json",
        ROOT / "scripts/run_statsbomb360_r12.py",
        ROOT / "results/statsbomb360/source_manifest.csv",
    ]
    files += sorted((ROOT / "results/statsbomb360_r12a").glob("*"))
    files = [p for p in files if p.is_file()]
    record = {
        "source_commit": "8799ab4cd236178a07372918e30ce450b54792d6",
        "input_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in files},
        "frozen_action_counts": {"r12_distributions": 4831, "exact_xr_uuid_joins": 4802, "matches": 115},
        "models": {
            "M0": ["frozen_rho_no_fit"],
            "M1_pooled": ["logit_rho", "competition"],
            "M1_competition": ["logit_rho"],
            "M2_pooled": ["logit_rho", "under_pressure", "goal_kick", "competition"],
            "M2_competition": ["logit_rho", "under_pressure", "goal_kick"],
            "M3_increment": ["lane_clear_short_options", "central_short_exit"],
            "M4_increment": ["receiver_option_count"],
            "M2C_increment": ["visible_area_fraction_of_pitch"],
            "component_ablations": ["lane_clear_short_options", "central_short_exit"],
        },
        "primary_attribution": "M3_minus_M2",
        "camera_robustness": "M3C_minus_M2C",
        "folds": "5-fold GroupKFold by match, identical assignment within scope",
        "bootstrap": "5000 paired match-cluster draws; retain match multiplicity; tie-aware AUC",
        "option_definitions": "unchanged from R1.2",
    }
    path = OUT / "input_and_model_lock.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != record:
            raise RuntimeError("Frozen input/model lock would change")
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"R1.2B inputs frozen: {len(files)} files")


if __name__ == "__main__":
    main()
