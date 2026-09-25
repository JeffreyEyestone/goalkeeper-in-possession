from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32a'
req=['calibration_transport.csv','calibration_learning_repeats.csv','calibration_learning_summary.csv','loco_competition_predictions.parquet','loco_competition_metrics.csv','calibration_impact_xtgk.csv','context_known_environment.csv','context_known_environment_deltas.csv','unseen_team_predictions.parquet','unseen_team_metrics.csv','contextual_fit.csv','keeper_context_standardization.csv']
def main():
 miss=[f for f in req if not (R/f).exists()]; assert not miss,miss
 l=pd.read_csv(R/'calibration_learning_repeats.csv'); assert len(l)>0 and l.groupby(['method','calibration_unit','calibration_size']).repeat.nunique().min()>=100
 assert l[['gap','brier','log_loss','ece']].notna().all().all()
 c=pd.read_csv(R/'calibration_transport.csv'); assert {'strict_zero_shot','local_intercept_only','local_intercept_slope','local_isotonic'}<=set(c.method)
 f=pd.read_csv(R/'contextual_fit.csv'); assert f.target_cohort.nunique()==5 and f.coverage.between(0,1).all() and f.unsupported_mass.between(0,1).all() and (f.effective_sample_size>0).any()
 assert len(pd.read_parquet(R/'loco_competition_predictions.parquet'))>0 and len(pd.read_parquet(R/'unseen_team_predictions.parquet'))>0
 print('R3.2A verification PASS')
if __name__=='__main__':main()
