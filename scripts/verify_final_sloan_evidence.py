from pathlib import Path
import json, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/final_evidence'
def main():
 req=['claim_ledger.csv','deep_consequence_final.csv','deep_consequence_bootstrap.csv','completion_value_final.csv','repeatability_final.csv','tf_ta_geo_final.csv','tf_ta_geo_bootstrap.csv','receiver_supporting_final.csv','context_final.csv','restart_claim_final.csv','abstract_fact_set.json','paper_fact_set.json','figure_plan.csv']
 assert all((R/f).exists() and (R/f).stat().st_size>0 for f in req)
 led=pd.read_csv(R/'claim_ledger.csv'); assert (led.status!='FAIL').any(); facts=json.load(open(R/'abstract_fact_set.json')); assert all(f['status'] in {'PASS','QUALIFIED'} and f['canonical_source'] and f['generating_code'] and f['git_commit'] for f in facts)
 assert len(pd.read_csv(R/'deep_consequence_bootstrap.csv'))>=50000
 assert len(pd.read_csv(R/'tf_ta_geo_bootstrap.csv'))>=10000
 assert set(pd.read_csv(R/'tf_ta_geo_final.csv').n)=={len(pd.read_parquet(ROOT/'results/r32b_closeout/receiver_predictions.parquet'))}
 assert not any('profile' in str(f).lower() or '#6' in str(f) or 'long-target' in str(f).lower() for f in facts)
 fig=pd.read_csv(R/'figure_plan.csv'); assert not fig.topic.str.contains('40|#6|long',case=False).any()
 rep=(ROOT/'reports/FINAL_SLOAN_EVIDENCE_LOCK.md').read_text(); assert 'Claims that died' in rep and 'DO-NOT-USE' in rep and '40–60' in rep
 print('FINAL SLOAN EVIDENCE verification PASS')
if __name__=='__main__':main()
