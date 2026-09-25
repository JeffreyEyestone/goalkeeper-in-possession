from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32b'; S=(ROOT/'scripts/r32b.py').read_text()
def main():
 req=['receiver_availability.csv','receiver_role_mapping.csv','receiver_role_coverage.csv','receiver_model_predictions.parquet','receiver_model_metrics.csv','receiver_incremental_ci.csv','known_receiver_metrics.csv','unseen_receiver_predictions.parquet','unseen_receiver_metrics.csv','unseen_team_receiver_predictions.parquet','unseen_team_receiver_metrics.csv','receiver_profile_coverage.csv','receiver_profiles.parquet','gk_receiver_adjustment.csv','gk_receiver_rank_changes.csv','pair_effects.csv','pair_model_comparison.csv','central_6_proxy_actions.parquet','central_6_proxy_results.csv','central_6_receiver_effects.csv','long_target_proxy_actions.parquet','long_target_proxy_results.csv','long_target_receiver_effects.csv','restart_intended_distance.csv','restart_distance_curve.csv','restart_receiver_conditioning.csv','paper_relevance_decision.csv','experiment_provenance.csv']
 assert all((R/f).exists() and (R/f).stat().st_size>0 for f in req)
 assert 'positions.update' not in S and 'OTHER_UNKNOWN' in S and all(k in S for k in ['CB','FB_WB','DM_6','CM_8','AM_10','WINGER','STRIKER'])
 assert 'receiver_availability' in S and set(pd.read_csv(R/'receiver_incremental_ci.csv').bootstrap_draws.unique())=={5000}
 assert len(pd.read_csv(R/'receiver_incremental_ci.csv'))>=25
 assert 'receiver_profile' in S and 'train' in S
 up=pd.read_parquet(R/'unseen_receiver_predictions.parquet'); assert len(up)>0
 ut=pd.read_parquet(R/'unseen_team_receiver_predictions.parquet'); assert len(ut)>0
 assert 'x.team!=team' in S
 assert 'groupby([\'keeper_id\',\'receiver_id\',\'cohort\'])' in S
 c6=pd.read_parquet(R/'central_6_proxy_actions.parquet'); assert 'DM_6' in S
 long=pd.read_parquet(R/'long_target_proxy_actions.parquet'); assert 'STRIKER' in S
 assert 'target_distance_m' in S and 'realized_distance_m' not in S[S.index('rk=x[x.goal_kick==1]'):]
 for f in ['R32B_SCIENCE_LOCK.md','R32B_RECEIVER_MODEL.md','R32B_RECEIVER_GENERALIZATION.md','R32B_GK_RECEIVER_CREDIT.md','R32B_PAIR_EFFECT.md','R32B_CENTRAL_6_PROXY.md','R32B_LONG_TARGET_FORWARD.md','R32B_RESTARTS.md','R32B_TRACKING_FUTURE.md']:
  p=ROOT/'reports'/f; assert p.exists() and '|' in p.read_text()
 print('R3.2B verification PASS')
if __name__=='__main__':main()
