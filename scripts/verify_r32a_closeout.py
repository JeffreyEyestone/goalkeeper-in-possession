from pathlib import Path
import re, json, hashlib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r32a_closeout'; S=(ROOT/'scripts/r32a_closeout.py').read_text()
import sys; sys.path.insert(0,str(ROOT/'src'))
def main():
 req=['xtgk_action_values.parquet','xtgk_action_sensitivity.csv','keeper_values.csv','keeper_rank_sensitivity.csv','loco_provenance.csv','team_context_bootstrap.csv','team_context_bootstrap_summary.csv']
 assert all((R/f).exists() for f in req) and (ROOT/'config/r32a_closeout_calibration_matches.json').exists()
 assert 'success_payoff_TF-ev.origin_value_TF' not in S and 'rho*success' not in S
 assert 'score_components' in S
 cfg=json.loads((ROOT/'config/r32a_closeout_calibration_matches.json').read_text()); assert cfg['seed']==42 and len(cfg['matches'])==5
 assert 'default_rng(42)' in S
 av=pd.read_parquet(R/'xtgk_action_values.parquet'); from gkpossession.xtgk import score_components
 # action-by-action canonical equality for every regime
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet').merge(pd.read_parquet(ROOT/'results/r32a_final/loco_predictions.parquet')[['event_id','p_loco']],on='event_id')
 for r in ['R0','R1','R2','R3']:
  assert np.allclose(av['xt_'+r], score_components(av[r],v.set_index('event_id').loc[av.event_id,'success_payoff_TF'].to_numpy(),v.set_index('event_id').loc[av.event_id,'origin_value_TF'].to_numpy(),v.set_index('event_id').loc[av.event_id,'failure_cost_TF'].to_numpy()))
 p=pd.read_csv(R/'loco_regeneration_check.csv'); assert p.max_abs_difference.max()<1e-12
 prov=pd.read_csv(R/'loco_provenance.csv'); assert prov.model_sha256.notna().all() and prov.calibrator_sha256.notna().all() and prov.model_sha256.nunique()==5 and (prov.beta!=1).any()
 for f,h in zip(prov.model_file,prov.model_sha256): assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 for f,h in zip(prov.calibrator_file,prov.calibrator_sha256): assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 boot=pd.read_csv(R/'team_context_bootstrap.csv'); assert len(boot)>=10000
 kv=pd.read_csv(R/'keeper_values.csv');
 for c,g in kv.groupby('cohort'):
  lim=25 if c in {'WC2018','WC2022','T2024'} else 200; assert (g.n_actions>=lim).all();
  if c in {'WC2018','WC2022','T2024'}: assert (g.n_matches>=2).all()
 assert len(list((ROOT/'reports').glob('R32A_*CLOSEOUT.md')))==2 and (ROOT/'reports/R32A_CONTEXT_SCIENCE_LOCK.md').exists()
 print('R3.2A CLOSEOUT verification PASS')
if __name__=='__main__': main()
