from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_abstract_facts_are_admissible():
 facts=json.load(open(ROOT/'results/final_evidence/abstract_fact_set.json')); assert all(f['status'] in {'PASS','QUALIFIED'} for f in facts); assert not any('profile' in f['text_label'].lower() for f in facts)
def test_deep_bootstrap_and_lock_report_exist():
 assert len(pd.read_csv(ROOT/'results/final_evidence/deep_consequence_bootstrap.csv'))>=50000; assert (ROOT/'reports/FINAL_SLOAN_EVIDENCE_LOCK.md').exists()
