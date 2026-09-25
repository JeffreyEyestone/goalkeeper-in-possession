PYTHON ?= python3.12
VENV ?= .venv312
PY = $(VENV)/bin/python
export PYTHONHASHSEED = 42
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
.PHONY: setup data test audit label-study xr values context evidence r2 receiver r3 r32a r32a-final r32a-patch r32a-closeout r32b r32b-closeout final-evidence v2-evidence reproduce paper-results
setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements.lock
data:
	$(PY) scripts/download_data.py
test:
	$(PY) -m pytest -q
audit: data
	$(PY) scripts/audit_metadata.py
	$(PY) scripts/historical_audit.py
	$(PY) scripts/audit_semantics.py
	$(PY) scripts/write_semantics_report.py
label-study: data
	$(PY) scripts/build_research_data.py
	$(PY) scripts/label_decision.py
xr: label-study
	$(PY) scripts/fit_xr.py
	$(PY) scripts/analyze_xr.py
	$(PY) scripts/report_xr.py
values: xr
	$(PY) scripts/build_surface_data.py
	$(PY) scripts/fit_values.py
	$(PY) scripts/report_values.py
context: values
	$(PY) scripts/contextual_analysis.py
evidence: context
	$(PY) scripts/context_evidence_gate.py
r2: evidence
	$(PY) scripts/correction_r2.py
receiver: r2
	$(PY) scripts/receiver_extension.py
r3: receiver
	$(PY) scripts/r3_lock.py
r32a: r3
	$(PY) scripts/r32a_finalize.py
	$(PY) scripts/verify_r32a.py
r32a-final: r3
	$(PY) scripts/r32a_final.py
	$(PY) scripts/verify_r32a_final.py
reproduce: r32a
	$(PY) scripts/r31_finalize.py
	$(PY) scripts/verify_r31_completion.py
	$(PY) -m pytest -q
paper-results:
	$(PY) scripts/scientific_gate.py
r32a-patch: r32a-final
	$(PY) scripts/r32a_patch.py
	$(PY) scripts/write_r32a_patch_reports.py
	$(PY) scripts/verify_r32a_patch.py

r32a-closeout: r32a-patch
	$(PY) scripts/r32a_closeout.py
	$(PY) scripts/write_r32a_closeout_reports.py
	$(PY) scripts/verify_r32a_closeout.py
r32b: r32a-closeout
	$(PY) scripts/r32b.py
	$(PY) scripts/write_r32b_reports.py
	$(PY) scripts/verify_r32b.py

r32b-closeout: r32b
	$(PY) scripts/r32b_closeout_repair.py
	$(PY) scripts/write_r32b_closeout_reports.py
	$(PY) scripts/verify_r32b_closeout.py

final-evidence: r32b-closeout
	$(PY) scripts/final_evidence.py
	$(PY) scripts/write_final_sloan_report.py
	$(PY) scripts/verify_final_sloan_evidence.py

v2-evidence: final-evidence
	$(PY) scripts/final_evidence_v2.py
	$(PY) scripts/write_final_sloan_v2.py
	$(PY) scripts/verify_final_evidence_v2.py
