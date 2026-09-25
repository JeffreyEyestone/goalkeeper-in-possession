from pathlib import Path
import json,pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/final_evidence_v2'; S=(ROOT/'scripts/final_evidence_v2.py').read_text()
def main():
 req=['tf_ta_geo_by_cohort.csv','tf_ta_geo_bootstrap_summary.csv','tf_ta_geo_bootstrap_draws.csv','completion_value_final.csv','completion_value_bootstrap.csv','repeatability_final.csv','repeatability_bootstrap.csv','deep_consequence_summary.csv','receiver_supporting_final.csv','context_final.csv','restart_claim_final.csv','claim_ledger.csv','abstract_fact_set.json','paper_fact_set.json','figure_plan.csv']
 assert all((R/f).exists() and (R/f).stat().st_size>0 for f in req)
 assert 'np.concatenate([groups[i] for i in draw])' in S and 'n=10000' in S
 tf=pd.read_csv(R/'tf_ta_geo_by_cohort.csv'); assert set(tf.cohort)=={'PL1516','T2024','WC2018','WC2022','WSL2021','OVERALL'}
 assert len(pd.read_csv(R/'tf_ta_geo_bootstrap_draws.csv'))>=50000
 cv=pd.read_csv(R/'completion_value_final.csv'); assert all((cv[cv.cohort.isin(['PL1516','WSL2021'])].eligibility_actions>=200)) and all((cv[cv.cohort.isin(['WC2018','WC2022','T2024'])].eligibility_actions>=25)); assert pd.read_csv(R/'completion_value_bootstrap.csv').shape[0]>=50000
 rep=pd.read_csv(R/'repeatability_final.csv'); assert set(rep.variable)>={'complete','R1','rho_TF','rho_TA_GEO','TA_residual'} and 'spearman_brown' in rep and 'hash(' not in S
 led=pd.read_csv(R/'claim_ledger.csv'); assert set(['C3','C9']).issubset(set(led.claim_id)); facts=json.load(open(R/'abstract_fact_set.json')); assert all(f['status'] in {'PASS','QUALIFIED'} and f['source_commit']!='FINAL_EVIDENCE_LOCK_SOURCE' for f in facts); assert not any(any(x in f['text_label'].lower() for x in ['profile','#6','long','pair','40']) for f in facts)
 fig=pd.read_csv(R/'figure_plan.csv'); assert len(fig)<=5 and not fig.topic.str.contains('receiver|40|#6|long',case=False).any()
 report=(ROOT/'reports/FINAL_SLOAN_EVIDENCE_LOCK_V2.md').read_text(); assert 'Claims that failed' in report and 'DO-NOT-USE' in report
 print('FINAL EVIDENCE V2 verification PASS')
if __name__=='__main__':main()
