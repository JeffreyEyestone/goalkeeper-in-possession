"""R3.2A Science Closeout: canonical xT-GK and provenance reconciliation."""
from pathlib import Path
import sys, hashlib, json
import numpy as np, pandas as pd, joblib
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gkpossession.stats import calibration
from gkpossession.calibration import intercept_only
from gkpossession.xtgk import score_components
OUT=ROOT/'results/r32a_closeout'; OUT.mkdir(exist_ok=True); MOD=ROOT/'models/r32a_closeout'; MOD.mkdir(parents=True,exist_ok=True)
NUM=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference']; CAT=['body_part']
def model():
 pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),NUM),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
 return Pipeline([('p',pre),('m',LogisticRegression(max_iter=500,C=.7,random_state=42))])
def sha_ids(v): return hashlib.sha256('|'.join(map(str,sorted(set(v)))).encode()).hexdigest()
def main():
 x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet'); x=x[x.R1.notna()].copy().reset_index(drop=True)
 accepted=pd.read_parquet(ROOT/'results/r32a_final/loco_predictions.parquet')[['event_id','p_loco']]; x=x.merge(accepted,on='event_id',how='inner')
 prov=[]; pcheck=[]
 for cohort,g in x.groupby('cohort',sort=True):
  train=x[x.cohort!=cohort]; ids=np.array(sorted(train.match_id.unique())); rng=np.random.default_rng(42); rng.shuffle(ids); calids=ids[:max(1,len(ids)//4)]; baseids=ids[max(1,len(ids)//4):]; tr=train[train.match_id.isin(baseids)]; ca=train[train.match_id.isin(calids)]
  m=model(); m.fit(tr[NUM+CAT],tr.R1.astype(int)); raw=m.predict_proba(g[NUM+CAT])[:,1]; a,b=calibration(ca.R1.astype(int),m.predict_proba(ca[NUM+CAT])[:,1]); regenerated=expit(a+b*logit(np.clip(raw,1e-5,1-1e-5)))
  mf=MOD/f'LOCO_{cohort}.joblib'; joblib.dump(m,mf); cf=MOD/f'CALIBRATOR_{cohort}.json'; cf.write_text(json.dumps({'alpha':float(a),'beta':float(b),'calibration_model':'intercept+slope','feature_registry':NUM+CAT},sort_keys=True))
  pcheck.append({'cohort':cohort,'max_abs_difference':float(np.max(np.abs(regenerated-g.p_loco.to_numpy()))),'regenerated_matches':len(regenerated)})
  prov.append({'held_out_cohort':cohort,'model_file':str(mf.relative_to(ROOT)),'model_sha256':hashlib.sha256(mf.read_bytes()).hexdigest(),'calibrator_file':str(cf.relative_to(ROOT)),'calibrator_sha256':hashlib.sha256(cf.read_bytes()).hexdigest(),'alpha':a,'beta':b,'training_match_hash':sha_ids(tr.match_id),'calibration_match_hash':sha_ids(ca.match_id),'held_out_match_hash':sha_ids(g.match_id)})
 pd.DataFrame(prov).to_csv(OUT/'loco_provenance.csv',index=False); pd.DataFrame(pcheck).to_csv(OUT/'loco_regeneration_check.csv',index=False)
 # Freeze seed-42 local calibration matches by cohort.
 split={}
 for cohort,g in x.groupby('cohort',sort=True):
  ids=np.array(sorted(g.match_id.unique())); rng=np.random.default_rng(42); rng.shuffle(ids); split[cohort]=[str(v) for v in ids[:min(5,len(ids))]]
 (ROOT/'config/r32a_closeout_calibration_matches.json').write_text(json.dumps({'seed':42,'matches':split},indent=2,sort_keys=True)+'\n')
 # Canonical four-regime xT on common evaluation actions.
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet').merge(x[['event_id','p_loco']],on='event_id',how='inner'); acts=[]; sens=[]; kval=[]; shifts=[]
 for cohort,g in v.groupby('cohort',sort=True):
  mids=split[cohort]; ca=g[g.match_id.astype(str).isin(mids)]; ev=g[~g.match_id.astype(str).isin(mids)].copy(); pca=ca.p_loco.to_numpy(); a=intercept_only(ca.R1.to_numpy(),pca); aa,bb=calibration(ca.R1.astype(int),pca)
  ev['R0']=ev.rho_TF; ev['R1']=ev.p_loco; ev['R2']=expit(logit(np.clip(ev.p_loco,1e-5,1-1e-5))+a); ev['R3']=expit(aa+bb*logit(np.clip(ev.p_loco,1e-5,1-1e-5)))
  for r in ['R0','R1','R2','R3']: ev['xt_'+r]=score_components(ev[r],ev.success_payoff_TF,ev.origin_value_TF,ev.failure_cost_TF)
  for u,w in [('R0','R1'),('R1','R2'),('R1','R3')]:
   d=(ev['xt_'+u]-ev['xt_'+w]).abs(); sens.append({'cohort':cohort,'comparison':u+'_vs_'+w,'evaluation_n':len(ev),'pearson':ev['xt_'+u].corr(ev['xt_'+w]),'spearman':ev['xt_'+u].corr(ev['xt_'+w],method='spearman'),'mean_abs_change':d.mean(),'median_abs_change':d.median()})
  tournament=cohort in {'WC2018','WC2022','T2024'}; threshold=25 if tournament else 200
  for regime in ['R0','R1','R2','R3']:
   q=ev.groupby(['keeper_id','keeper']).agg(n_actions=('xt_'+regime,'size'),n_matches=('match_id','nunique'),value=('xt_'+regime,'mean')).reset_index(); q=q[q.n_actions>=threshold].copy(); q=q[q.n_matches>=2] if tournament else q; q['cohort']=cohort; q['regime']=regime; q['rank']=q.value.rank(ascending=False,method='average'); kval.append(q)
  acts.append(ev[['event_id','cohort','keeper_id','keeper','match_id','R0','R1','R2','R3','xt_R0','xt_R1','xt_R2','xt_R3']])
 av=pd.concat(acts); av.to_parquet(OUT/'xtgk_action_values.parquet',index=False); pd.DataFrame(sens).to_csv(OUT/'xtgk_action_sensitivity.csv',index=False); kv=pd.concat(kval); kv.to_csv(OUT/'keeper_values.csv',index=False)
 for (c,kid,name),q in kv.groupby(['cohort','keeper_id','keeper']):
  b=q[q.regime=='R1'].iloc[0]
  for reg in ['R0','R2','R3']:
   z=q[q.regime==reg].iloc[0]; shifts.append({'cohort':c,'keeper_id':kid,'keeper':name,'regime':reg,'n_actions':int(z.n_actions),'n_matches':int(z.n_matches),'rank_shift':abs(float(z['rank']-b['rank']))})
 sh=pd.DataFrame(shifts); sh.to_csv(OUT/'keeper_rank_sensitivity.csv',index=False)
 # team bootstrap from accepted paired results (descriptive, 10,000 draws)
 t=pd.read_csv(ROOT/'results/r32a_patch/unseen_team_competition_metrics.csv'); rng=np.random.default_rng(42); rows=[]
 for i in range(10000):
  q=t.iloc[rng.integers(0,len(t),len(t))]; rows.append({'draw':i,'mean_delta_auc':q.delta_AUC.mean(),'median_delta_auc':q.delta_AUC.median(),'mean_delta_brier':q.delta_Brier.mean(),'median_delta_brier':q.delta_Brier.median(),'median_abs_gap_change':(q.gap_B.abs()-q.gap_A.abs()).median(),'median_ece_change':(q.ECE_B-q.ECE_A).median()})
 boot=pd.DataFrame(rows); boot.to_csv(OUT/'team_context_bootstrap.csv',index=False); boot.quantile([.025,.5,.975]).to_csv(OUT/'team_context_bootstrap_summary.csv')
 print('R3.2A Science Closeout outputs written')
if __name__=='__main__': main()
