from pathlib import Path
import re, hashlib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32a_patch'; S=(ROOT/'scripts/r32a_patch.py').read_text(); F=(ROOT/'scripts/r32a_final.py').read_text()
def main():
 req=['loco_zero_shot.csv','loco_local_calibration_repeats.csv','loco_local_calibration_summary.csv','loco_provenance.csv','unseen_team_competition_metrics.csv','unseen_team_competition_predictions.parquet','unseen_team_competition_summary.csv','xtgk_patch_action_values.parquet','xtgk_patch_propagation.csv','xtgk_patch_keeper_ranks.csv','xtgk_patch_keeper_rank_shifts.csv']
 assert all((R/f).exists() for f in req)
 assert 'mean(y)-mean(p)' not in F and 'def intercept_only' not in F
 assert 'g.p_loco' in S and 'rho_TF' not in S[S.index('for k in [1,3,5]'):S.index('# Save five')]
 lr=pd.read_csv(R/'loco_local_calibration_repeats.csv'); assert set(lr.method)=={'intercept_only','intercept_slope'} and len(lr)>0
 assert not np.allclose(lr[lr.method=='intercept_only'].calibration_slope,lr[lr.method=='intercept_slope'].calibration_slope)
 tm=pd.read_csv(R/'unseen_team_competition_metrics.csv'); assert len(tm)>0 and {'AUC_A','AUC_B','delta_AUC'}.issubset(tm.columns)
 pred=pd.read_parquet(R/'unseen_team_competition_predictions.parquet'); assert not np.allclose(pred.p_A,pred.p_B)
 prov=pd.read_csv(R/'loco_provenance.csv'); assert prov.model_sha256.nunique()==5 and prov.model_file.notna().all() and prov.calibrator_sha256.notna().all()
 for f in prov.model_file: assert (ROOT/f).exists()
 xr=pd.read_parquet(R/'xtgk_patch_action_values.parquet'); assert {'xt_R0','xt_R1','xt_R2','xt_R3'}.issubset(xr.columns)
 rk=pd.read_csv(R/'xtgk_patch_keeper_ranks.csv'); assert rk.groupby('cohort').ngroups==5
 assert len(list((ROOT/'reports').glob('R32A_PATCH_*.md')))==4
 z=pd.read_csv(R/'loco_zero_shot.csv'); assert len(z)==5
 print('R3.2A PATCH verification PASS')
if __name__=='__main__':main()
