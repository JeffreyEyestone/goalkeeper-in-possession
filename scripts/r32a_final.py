"""True LOCO, leave-one-team-out, and xT-GK recalibration propagation."""
from pathlib import Path
import sys
import hashlib, json, numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from scipy.special import expit, logit
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/r32a_final'; OUT.mkdir(exist_ok=True)
from gkpossession.stats import calibration
from gkpossession.calibration import intercept_only
NUM=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference']; CAT=['body_part']
def sha(vals): return hashlib.sha256('|'.join(map(str,sorted(set(vals)))).encode()).hexdigest()
def model(cats=CAT):
 pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),NUM),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cats)])
 return Pipeline([('p',pre),('m',LogisticRegression(max_iter=500,C=.7,random_state=42))])
def score(y,p):
 b=np.minimum((p*10).astype(int),9); a,s=calibration(y,p); return {'n':len(y),'prevalence':float(np.mean(y)),'mean_prediction':float(np.mean(p)),'gap':float(np.mean(p)-np.mean(y)),'auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]),'ece':float(sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any())),'calibration_intercept':a,'calibration_slope':s}
def main():
 x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet'); x=x[x.R1.notna()].copy().reset_index(drop=True); preds=[]; mets=[]; prov=[]
 for hold,g in x.groupby('cohort'):
  train=x[x.cohort!=hold]; ids=np.array(sorted(train.match_id.unique())); rng=np.random.default_rng(42); rng.shuffle(ids); calids=ids[:max(1,len(ids)//4)]; baseids=ids[max(1,len(ids)//4):]; tr=train[train.match_id.isin(baseids)]; ca=train[train.match_id.isin(calids)]; m=model(); m.fit(tr[NUM+CAT],tr.R1.astype(int)); p0=m.predict_proba(g[NUM+CAT])[:,1]; cal=model(); cal.fit(tr[NUM+CAT],tr.R1.astype(int)); raw=cal.predict_proba(ca[NUM+CAT])[:,1]; a,b=calibration(ca.R1.astype(int),raw); p=expit(a+b*logit(np.clip(p0,1e-5,1-1e-5))); h=hashlib.sha256(repr(m).encode()).hexdigest(); preds.append(pd.DataFrame({'event_id':g.event_id,'match_id':g.match_id,'cohort':hold,'y':g.R1.astype(int),'p_loco':p,'p_frozen':g.rho_TF})); mets.append({'held_out_cohort':hold,**score(g.R1.astype(int),p)}); prov.append({'model_id':'LOCO_'+hold,'model_hash':h,'feature_registry_hash':hashlib.sha256('|'.join(NUM+CAT).encode()).hexdigest(),'held_out_cohort':hold,'training_cohorts':','.join(sorted(train.cohort.unique())),'training_match_ids_hash':sha(tr.match_id),'calibration_match_ids_hash':sha(ca.match_id),'test_match_ids_hash':sha(g.match_id),'training_actions':len(tr),'calibration_actions':len(ca),'test_actions':len(g),'random_seed':42})
  # local calibration scenarios on held-out cohort, repeated draws.
  for k in [1,3,5]:
   for rep in range(100):
    mids=np.random.default_rng(rep).choice(g.match_id.unique(),min(k,g.match_id.nunique()),replace=False); lc=g[g.match_id.isin(mids)]; ev=g[~g.match_id.isin(mids)]
    if len(ev)<25: continue
    aa,_=calibration(lc.R1.astype(int),lc.rho_TF); pp=expit(logit(np.clip(ev.rho_TF,1e-5,1-1e-5))+aa); mets.append({'held_out_cohort':hold,'scenario':f'local_{k}_matches','repeat':rep,**score(ev.R1.astype(int),pp)})
 pd.concat(preds).to_parquet(OUT/'loco_predictions.parquet',index=False); pd.DataFrame(mets).to_csv(OUT/'loco_metrics.csv',index=False); pd.DataFrame(prov).to_csv(OUT/'loco_provenance.csv',index=False); pd.DataFrame([{'held_out_cohort':r['held_out_cohort'],'scenario':'local','calibration_matches':r.get('scenario','')} for r in mets if 'scenario' in r]).to_csv(OUT/'loco_local_calibration.csv',index=False)
 # Leave-one-team-out, selected teams with >=250 actions.
 tm=[]; tp=[]; tprov=[]
 for tid,g in x.groupby('team_id'):
  if len(g)<250: continue
  train=x[x.team_id!=tid]; m=model(); m.fit(train[NUM+CAT],train.R1.astype(int)); p=m.predict_proba(g[NUM+CAT])[:,1]; tp.append(pd.DataFrame({'event_id':g.event_id,'match_id':g.match_id,'team_id':tid,'y':g.R1.astype(int),'p_global':p,'p_frozen':g.rho_TF})); tm.append({'held_out_team_id':tid,'held_out_team_name':g.team.iloc[0],**score(g.R1.astype(int),p)}); tprov.append({'held_out_team_id':tid,'held_out_team_name':g.team.iloc[0],'training_team_ids_hash':sha(train.team_id),'training_match_ids_hash':sha(train.match_id),'model_hash':hashlib.sha256(repr(m).encode()).hexdigest(),'feature_registry_hash':hashlib.sha256('|'.join(NUM+CAT).encode()).hexdigest(),'training_actions':len(train),'test_actions':len(g),'competition':','.join(sorted(g.cohort.unique())),'seed':42})
 pd.concat(tp).to_parquet(OUT/'unseen_team_predictions.parquet',index=False); pd.DataFrame(tm).to_csv(OUT/'unseen_team_metrics.csv',index=False); pd.DataFrame(tm).describe(percentiles=[.1,.5,.9]).T.to_csv(OUT/'unseen_team_summary.csv'); pd.DataFrame(tprov).to_csv(OUT/'unseen_team_provenance.csv',index=False)
 # xT propagation: fixed branch functions from valued_actions, probability regime only changes.
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); pmap=pd.concat(preds).set_index('event_id'); v=v[v.event_id.isin(pmap.index)].copy().set_index('event_id'); v['rho_strict']=v.rho_TF; v['rho_loco']=pmap.p_loco; v['rho_intercept']=v.rho_TF
 for c,g in v.groupby('cohort'):
  mids=sorted(g.match_id.unique()); lc=g[g.match_id==mids[0]]; alpha=intercept_only(lc.R1.astype(int),lc.rho_TF); v.loc[g.index,'rho_intercept']=expit(logit(np.clip(g.rho_TF,1e-5,1-1e-5))+alpha)
 success=v.success_payoff_TF-v.origin_value_TF; failure=v.failure_cost_TF; 
 for col in ['rho_strict','rho_loco','rho_intercept']: v['xt_'+col]=v[col]*success-(1-v[col])*failure
 v.reset_index().to_parquet(OUT/'xtgk_action_values.parquet',index=False)
 rows=[]
 for c,g in v.groupby('cohort'):
  for a,b in [('strict','loco'),('strict','intercept')]: rows.append({'cohort':c,'comparison':a+'_vs_'+b,'pearson':g['xt_rho_'+a].corr(g['xt_rho_'+b]),'spearman':g['xt_rho_'+a].corr(g['xt_rho_'+b],method='spearman'),'mean_abs_value_change':(g['xt_rho_'+a]-g['xt_rho_'+b]).abs().mean()})
 pd.DataFrame(rows).to_csv(OUT/'xtgk_propagation_metrics.csv',index=False)
 k=v.groupby(['keeper_id','keeper']).agg(n=('xt_rho_strict','size'),strict=('xt_rho_strict','mean'),loco=('xt_rho_loco','mean')).query('n>=100').reset_index(); k['strict_rank']=k.strict.rank(ascending=False); k['loco_rank']=k.loco.rank(ascending=False); k['rank_shift']=(k.strict_rank-k.loco_rank).abs(); k.to_csv(OUT/'xtgk_keeper_rank_changes.csv',index=False)
 print('R3.2A final experiments written',flush=True)
if __name__=='__main__':main()
