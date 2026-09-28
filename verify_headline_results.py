#!/usr/bin/env python3
"""Verify headline numerical claims in the minimal SSAC27 repository.

Uses only the Python standard library and the frozen publication evidence CSVs.
This script checks the public evidence package; it does not refit the models.
"""
from pathlib import Path
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
D = ROOT / "data"

errors = []

def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol

# Event portability
port = [r for r in rows("event_portability.csv") if r["cohort"] != "SUMMARY"]
aucs = [float(r["LOCO_AUC"]) for r in port]
if len(port) != 5 or min(aucs) < 0.644 or max(aucs) > 0.711:
    errors.append("Unexpected xR-GK LOCO AUC range")

# Failure asymmetry
fail = rows("failure_asymmetry.csv")
burdens = [float(r["V_plus_C_over_V"]) for r in fail]
if len(fail) != 5 or min(burdens) < 2.39 or max(burdens) > 2.91:
    errors.append("Unexpected complete lost-branch range")

# Target-aware geometry and selection caveat
geo = rows("target_aware_geometry.csv")
overall = next((r for r in geo if r["cohort"] == "OVERALL"), None)
if overall is None or not (0.18 < float(overall["delta_AUC"]) < 0.20):
    errors.append("Unexpected overall TA-GEO AUC gain")
sel = rows("target_selection.csv")
if len(sel) != 5 or not all(float(r["target_identifiable_pct_R1_1"]) > float(r["target_identifiable_pct_R1_0"]) for r in sel):
    errors.append("Target-identification selection pattern missing")

# Visible options and direct choice
choice = rows("option_choice_by_tournament.csv")
lookup = {(r["scope"], r["clear_short_options"]): float(r["direct_pct"]) for r in choice}
expected_choice = {
    ("WC2022","0"): 85.9375,
    ("WC2022","1"): 42.96875,
    ("WC2022","2+"): 27.179763186221745,
    ("EURO2024","0"): 91.92825112107623,
    ("EURO2024","1"): 57.14285714285714,
    ("EURO2024","2+"): 25.76985413290113,
}
for key, expected in expected_choice.items():
    if key not in lookup or abs(lookup[key]-expected) > 1e-9:
        errors.append(f"Unexpected direct-choice result for {key}")

# Primary visible-option retention attribution
primary = rows("visible_option_retention_validation.csv")
auc = next((r for r in primary if r["scope"]=="ALL" and r["metric"]=="AUC"), None)
if auc is None:
    errors.append("Missing pooled primary AUC comparison")
else:
    if not approx(auc["control_value"], 0.6484866502992175):
        errors.append("Unexpected pooled control AUC")
    if not approx(auc["augmented_value"], 0.699109372291051):
        errors.append("Unexpected pooled augmented AUC")
    if not approx(auc["delta"], 0.050622721991833486):
        errors.append("Unexpected pooled delta AUC")
    if not (0.034 < float(auc["ci_low"]) < 0.036 and 0.065 < float(auc["ci_high"]) < 0.067):
        errors.append("Unexpected pooled AUC confidence interval")

# Camera-visible-area robustness
camera = rows("camera_control_validation.csv")
cam_auc = next((r for r in camera if r["scope"]=="ALL" and r["metric"]=="AUC"), None)
if cam_auc is None or float(cam_auc["delta"]) <= 0 or float(cam_auc["ci_low"]) <= 0:
    errors.append("Camera-control robustness did not remain positive")

# Clean-source reproduction
repro = rows("source_reproduction_audit.csv")
if not repro or any(r["status"] != "MATCH" or int(r["discrepancies"]) != 0 for r in repro):
    errors.append("Raw-to-derived reproduction audit is not clean")
key_rows = [r for r in repro if r["field"]=="__key__"]
if sorted(int(r["frozen_rows"]) for r in key_rows) != [4831,27777]:
    errors.append("Frozen 360 row counts changed")

# Event cohort manifests
manifest_dir = D / "event" / "manifests"
counts = {}
for p in sorted(manifest_dir.glob("*.json")):
    values = json.loads(p.read_text())
    counts[p.stem] = len(values)
if sum(counts.values()) != 722:
    errors.append(f"Event match manifests total {sum(counts.values())}, expected 722")

# 360 source manifest
with (D/"360"/"source_manifest.csv").open(newline="", encoding="utf-8") as f:
    src = list(csv.DictReader(f))
if len(src) != 348 or any(int(r["bytes"]) <= 0 or r["http_status"] != "200" or r["json_valid"].lower() not in ("true","1") for r in src):
    errors.append("360 source manifest incomplete or invalid")
if len({r["official_commit"] for r in src}) != 1:
    errors.append("360 source files are not pinned to one upstream commit")

if errors:
    print("MINIMAL SLOAN REPO VERIFICATION: FAIL")
    for e in errors:
        print("-", e)
    sys.exit(1)

print("MINIMAL SLOAN REPO VERIFICATION: PASS")
print("Event matches:", sum(counts.values()))
print("360 official source files:", len(src))
print("xR-GK unseen-competition AUC range:", f"{min(aucs):.3f}-{max(aucs):.3f}")
print("Visible-option pooled AUC:", f"{float(auc['control_value']):.3f} -> {float(auc['augmented_value']):.3f}")
print("Complete lost-branch range:", f"{min(burdens):.2f}-{max(burdens):.2f}x")
