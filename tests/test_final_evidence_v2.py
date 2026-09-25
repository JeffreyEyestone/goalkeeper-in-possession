from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
def test_v2_cohort_tables_and_provenance():
 r=ROOT/'results/final_evidence_v2'; t=pd.read_csv(r/'tf_ta_geo_by_cohort.csv'); assert len(t)==6; f=json.load(open(r/'abstract_fact_set.json')); assert all(x['source_commit']!='FINAL_EVIDENCE_LOCK_SOURCE' for x in f)
def test_v2_repeatability_variables():
 v=pd.read_csv(ROOT/'results/final_evidence_v2/repeatability_final.csv'); assert {'complete','R1','rho_TF','rho_TA_GEO','TA_residual'}<=set(v.variable)
