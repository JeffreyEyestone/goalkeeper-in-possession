from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32b_closeout'; S=(ROOT/'scripts/r32b.py').read_text(); C=(ROOT/'scripts/r32b_closeout_repair.py').read_text()
def main():
 req=['raw_position_names.csv','receiver_role_mapping.csv','receiver_role_coverage.csv','receiver_predictions.parquet','receiver_model_metrics.csv','receiver_incremental_ci.csv','unseen_receiver_metrics.csv','unseen_team_receiver_metrics.csv','receiver_profiles.parquet','receiver_profile_coverage.csv','gk_receiver_adjustment.csv','gk_receiver_rank_changes.csv','pair_predictions.parquet','pair_model_comparison.csv','pair_effects.csv','central_6_actions.parquet','central_6_results.csv','central_6_receiver_effects.csv','long_target_actions.parquet','long_target_results.csv','long_target_receiver_effects.csv','restart_actions.parquet','restart_curve.csv','restart_bands.csv','restart_receiver_conditioning.csv','paper_relevance_decision.csv','experiment_provenance.csv']
 assert all((R/f).exists() and (R/f).stat().st_size>0 for f in req)
 rm=pd.read_csv(R/'receiver_role_mapping.csv'); vc=rm.receiver_role.value_counts(); assert vc.get('OTHER_UNKNOWN',0)<.95*len(rm) and all(vc.get(k,0)>0 for k in ['DM_6','STRIKER','CB','FB_WB'])
 assert 'pos.get(rid' in S and 'ROLE_MAP.get(pos.get(rid)' not in S
 pred=pd.read_parquet(R/'receiver_predictions.parquet'); assert not np.allclose(pred['p_TA-GEO'],pred['p_TA-ROLE']); pf=pd.read_csv(R/'receiver_profile_feature_check.csv'); assert pf.profile_in_preprocessor.all(); assert not np.allclose(pred['p_TA-GEO'],pred['p_TA-PROFILE'])
 ci=pd.read_csv(R/'receiver_incremental_ci.csv'); assert ci.bootstrap_draws.min()>=5000
 c6=pd.read_parquet(R/'central_6_actions.parquet'); lg=pd.read_parquet(R/'long_target_actions.parquet'); assert len(c6)>0 and len(lg)>0
 pair=pd.read_parquet(R/'pair_predictions.parquet'); assert not np.allclose(pair.p_base,pair.p_pair)
 curve=pd.read_csv(R/'restart_curve.csv'); assert curve.R1_prediction.notna().all() and curve.value_prediction.notna().all(); assert 'realized_distance_m' not in C
 for f in ['R32B_CLOSEOUT_SCIENCE_LOCK.md','R32B_CLOSEOUT_RECEIVERS.md','R32B_CLOSEOUT_CENTRAL_6.md','R32B_CLOSEOUT_LONG_TARGET.md','R32B_CLOSEOUT_RESTARTS.md','R32B_CLOSEOUT_PAIR_EFFECT.md']:
  p=ROOT/'reports'/f; assert p.exists() and '|' in p.read_text()
 print('R3.2B CLOSEOUT verification PASS')
if __name__=='__main__':main()
