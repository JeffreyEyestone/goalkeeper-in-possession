"""Re-execute only the frozen R1.2 option-table builder on clean raw data.

The original R1.2 generator is parsed and executed unchanged through its two
option-table writes. Later R1.2 regressions and reports are never executed.
Fresh results are written only to /private/tmp and compared field by field.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/statsbomb360_r12b_fresh_20260924")
TMP = Path("/private/tmp/statsbomb360_r12b_reproduction")
OUT = ROOT / "results/statsbomb360_r12b"
SOURCE = ROOT / "scripts/run_statsbomb360_r12.py"
SUMMARY_FIELDS = ["match_id", "competition", "goal_kick", "under_pressure",
                  "choice_family", "chosen_distance_m", "receiver_option_count",
                  "visible_short_options", "lane_clear_short_options",
                  "central_short_exit", "central_build_exit", "start_x", "start_y",
                  "end_x", "end_y", "retained", "chosen_receiver_id"]
CANDIDATE_FIELDS = ["match_id", "competition", "candidate_player_id",
                    "candidate_x_m", "candidate_y_m", "distance_m", "side",
                    "distance_family", "receiver_nearest_opponent_m",
                    "opponents_within_3m", "opponents_within_5m",
                    "lane_clear_1m", "lane_clear_1_5m", "lane_clear_2m",
                    "forward_progress_m", "lateral_displacement_m"]


def validate_manifest() -> None:
    manifest = pd.read_csv(OUT/"source_manifest_verified.csv")
    if len(manifest) != 348 or (manifest.bytes <= 0).any() or not manifest.json_valid.all() or not manifest.http_status.eq(200).all():
        raise RuntimeError("Fresh raw source manifest incomplete or invalid")
    if manifest.official_commit.nunique() != 1:
        raise RuntimeError("Raw sources not pinned to a single official commit")
    for row in manifest.itertuples():
        path = RAW/row.relative_path
        if not path.is_file() or path.stat().st_size != row.bytes:
            raise RuntimeError(f"Fresh raw file absent/size mismatch: {row.relative_path}")
        with path.open("rb") as file:
            json.load(file)


def execute_frozen_prefix() -> None:
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    stop = None
    for index, statement in enumerate(original.body):
        if "s.to_parquet(OUT / 'distribution_option_summary.parquet'" in ast.unparse(statement):
            stop = index + 1
            break
    if stop is None:
        raise RuntimeError("Could not isolate original frozen R1.2 option-table builder")
    function = copy.deepcopy(original)
    function.name = "build_frozen_option_tables"
    function.body = function.body[:stop]
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = runpy.run_path(str(SOURCE))
    namespace.update({"RAW": RAW, "OUT": TMP/"results/statsbomb360",
                      "FIG": TMP/"figures/statsbomb360", "REP": TMP/"reports"})
    namespace["OUT"].mkdir(parents=True, exist_ok=True)
    exec(compile(module, filename=str(SOURCE), mode="exec"), namespace)
    namespace["build_frozen_option_tables"]()


def compare_table(name: str, keys: list[str], fields: list[str]) -> list[dict]:
    frozen = pd.read_parquet(ROOT/"results/statsbomb360"/name)
    fresh = pd.read_parquet(TMP/"results/statsbomb360"/name)
    if frozen.duplicated(keys).any() or fresh.duplicated(keys).any():
        raise RuntimeError(f"Nonunique reproduction key in {name}")
    merged = frozen.merge(fresh, on=keys, how="outer", suffixes=("_frozen", "_fresh"),
                          indicator=True, validate="one_to_one")
    missing = int((merged._merge == "left_only").sum())
    added = int((merged._merge == "right_only").sum())
    common = merged[merged._merge == "both"]
    rows = [{"table": name, "field": "__key__", "frozen_rows": len(frozen),
             "fresh_rows": len(fresh), "matched_keys": len(common),
             "missing_frozen_keys": missing, "new_fresh_keys": added,
             "discrepancies": missing + added, "max_abs_difference": np.nan,
             "status": "MATCH" if missing + added == 0 else "DISCREPANCY"}]
    for field in fields:
        a = common[field+"_frozen"]
        b = common[field+"_fresh"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            left = a.to_numpy(dtype=float)
            right = b.to_numpy(dtype=float)
            equal = np.isclose(left, right, rtol=1e-10, atol=1e-8, equal_nan=True)
            diff = np.abs(left-right)
            max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
        else:
            equal = (a.fillna("<MISSING>").astype(str).to_numpy() ==
                     b.fillna("<MISSING>").astype(str).to_numpy())
            max_diff = np.nan
        discrepancies = int((~equal).sum())
        rows.append({"table": name, "field": field, "frozen_rows": len(frozen),
                     "fresh_rows": len(fresh), "matched_keys": len(common),
                     "missing_frozen_keys": missing, "new_fresh_keys": added,
                     "discrepancies": discrepancies, "max_abs_difference": max_diff,
                     "status": "MATCH" if discrepancies == 0 else "DISCREPANCY"})
    return rows


def main() -> None:
    validate_manifest()
    execute_frozen_prefix()
    rows = compare_table("distribution_option_summary.parquet", ["event_id"], SUMMARY_FIELDS)
    rows += compare_table("option_candidates.parquet", ["event_id", "candidate_index"], CANDIDATE_FIELDS)
    audit = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT/"raw_to_derived_reproduction.csv", index=False)
    if not audit.status.eq("MATCH").all():
        print(audit[audit.status != "MATCH"].to_string(index=False))
        raise RuntimeError("Fresh raw-to-derived reproduction has scientific-field discrepancies")
    print("Raw-to-derived reproduction: PASS; 4,831 distributions and 27,777 candidates matched")


if __name__ == "__main__":
    main()
