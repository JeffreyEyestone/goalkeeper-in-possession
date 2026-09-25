"""Respect approved TF/TA amendment and fail closed at remaining scientific gates."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1]
if not (root/'config/specifications/APPROVED_TF_TA_AMENDMENT.md').exists():sys.exit('STOP: target/estimand decision not approved.')
p=root/'results/v21/scientific_gates.json'
if not p.exists():sys.exit('STOP: run make xr before continuing beyond calibration.')
gates=json.loads(p.read_text())
if gates:sys.exit('SCIENTIFIC REVIEW REQUIRED: '+', '.join(f"{r['gate']} {r.get('estimand','')} {r.get('cohort','')}" for r in gates)+'. See reports/CALIBRATION_DECISION_REQUIRED.md.')
sys.exit('STOP: downstream valuation/validation is not yet complete; no final paper results are available.')
