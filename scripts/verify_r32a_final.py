from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32a_final'
def main():
 req=['loco_predictions.parquet','loco_metrics.csv','loco_local_calibration.csv','loco_provenance.csv','unseen_team_predictions.parquet','unseen_team_metrics.csv','unseen_team_summary.csv','unseen_team_provenance.csv','xtgk_action_values.parquet','xtgk_propagation_metrics.csv','xtgk_keeper_rank_changes.csv']
 assert all((R/f).exists() for f in req)
 lp=pd.read_parquet(R/'loco_predictions.parquet'); assert lp.cohort.nunique()==5 and not np.allclose(lp.p_loco,lp.p_frozen)
 pr=pd.read_csv(R/'loco_provenance.csv'); assert pr.model_hash.notna().all() and pr.training_match_ids_hash.notna().all()
 tp=pd.read_parquet(R/'unseen_team_predictions.parquet'); assert len(tp)>0 and {'p_global','p_frozen'}.issubset(tp.columns) and not np.allclose(tp.p_global,tp.p_frozen)
 xv=pd.read_parquet(R/'xtgk_action_values.parquet'); assert not np.allclose(xv.xt_rho_strict,xv.xt_rho_loco)
 k=pd.read_csv(R/'xtgk_keeper_rank_changes.csv'); assert len(k)>0
 print('R3.2A FINAL verification PASS')
if __name__=='__main__':main()
