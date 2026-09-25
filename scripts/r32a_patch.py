"""R3.2A Patch 1: canonical local calibration, competition context, and within-cohort xT-GK."""
from pathlib import Path
import sys, hashlib, json, shutil
import numpy as np, pandas as pd, joblib
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gkpossession.stats import calibration
from gkpossession.calibration import intercept_only
OUT=ROOT/'results/r32a_patch'; OUT.mkdir(exist_ok=True); MOD=ROOT/'models/r32a_patch'; MOD.mkdir(parents=True,exist_ok=True)
NUM=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference']; CAT=['body_part']

def model(cats=CAT):
 pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),NUM),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cats)])
 return Pipeline([('p',pre),('m',LogisticRegression(max_iter=500,C=.7,random_state=42))])
def metrics(y,p):
 b=np.minimum((p*10).astype(int),9); return {'n':len(y),'gap':float(p.mean()-y.mean()),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]),'ece':float(sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any())),'auc':roc_auc_score(y,p)}
def sha_ids(v): return hashlib.sha256('|'.join(map(str,sorted(set(v)))).encode()).hexdigest()
def fit_cal(y,p,method):
 if method=='intercept_only': a=intercept_only(y,p); b=1.0
 else: a,b=calibration(np.asarray(y).astype(int),np.asarray(p))
 return expit(a+b*logit(np.clip(np.asarray(p),1e-5,1-1e-5))),a,b
def main():
 x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet'); x=x[x.R1.notna()].copy().reset_index(drop=True)
 base=pd.read_parquet(ROOT/'results/r32a_final/loco_predictions.parquet')[['event_id','p_loco']]
 x=x.merge(base,on='event_id',how='inner'); zero=[]; local=[]; prov=[]
 for cohort,g in x.groupby('cohort',sort=True):
  # exact five-row zero-shot table comes from accepted LOCO predictions
  p=g.p_loco.to_numpy(); zero.append({'cohort':cohort,**metrics(g.R1.astype(int).to_numpy(),p)})
  ids=np.array(sorted(g.match_id.unique()));
  for k in [1,3,5]:
   if len(ids)<=k: continue
   for rep in range(100):
    rng=np.random.default_rng(42+rep+1000*k+hash(cohort)%997); mids=rng.choice(ids,k,replace=False); ca=g[g.match_id.isin(mids)]; ev=g[~g.match_id.isin(mids)]
    if len(ev)==0: continue
    for method in ['intercept_only','intercept_slope']:
     pp,a,b=fit_cal(ca.R1.to_numpy(),ca.p_loco.to_numpy(),method); pe=expit(a+b*logit(np.clip(ev.p_loco.to_numpy(),1e-5,1-1e-5))); mm=metrics(ev.R1.astype(int).to_numpy(),pe)
     local.append({'cohort':cohort,'method':method,'calibration_matches':k,'repeat':rep,'calibration_n':len(ca),'evaluation_n':len(ev),'calibration_intercept':a,'calibration_slope':b,**mm})
 # Save five LOCO model bytes and calibrators, with provenance.
 for cohort,g in x.groupby('cohort',sort=True):
  train=x[x.cohort!=cohort]; ids=np.array(sorted(train.match_id.unique())); rng=np.random.default_rng(42); rng.shuffle(ids); calids=ids[:max(1,len(ids)//4)]; baseids=ids[max(1,len(ids)//4):]; tr=train[train.match_id.isin(baseids)]; ca=train[train.match_id.isin(calids)]
  m=model(); m.fit(tr[NUM+CAT],tr.R1.astype(int)); mf=MOD/f'LOCO_{cohort}.joblib'; joblib.dump(m,mf); raw=mf.read_bytes(); cal=intercept_only(ca.R1.to_numpy(),m.predict_proba(ca[NUM+CAT])[:,1]); cf=MOD/f'CALIBRATOR_{cohort}.json'; cf.write_text(json.dumps({'alpha':cal,'beta':1.0})); prov.append({'held_out_cohort':cohort,'model_file':str(mf.relative_to(ROOT)),'model_sha256':hashlib.sha256(raw).hexdigest(),'calibrator_file':str(cf.relative_to(ROOT)),'calibrator_sha256':hashlib.sha256(cf.read_bytes()).hexdigest(),'training_cohorts':','.join(sorted(train.cohort.unique())),'training_match_hash':sha_ids(tr.match_id),'calibration_match_hash':sha_ids(ca.match_id),'held_out_match_hash':sha_ids(g.match_id)})
 pd.DataFrame(zero).to_csv(OUT/'loco_zero_shot.csv',index=False); lr=pd.DataFrame(local); lr.to_csv(OUT/'loco_local_calibration_repeats.csv',index=False)
 lr.groupby(['cohort','method','calibration_matches']).agg(repeats=('repeat','size'),gap_median=('gap','median'),gap_lo=('gap',lambda s:s.quantile(.025)),gap_hi=('gap',lambda s:s.quantile(.975)),brier_median=('brier','median'),brier_lo=('brier',lambda s:s.quantile(.025)),brier_hi=('brier',lambda s:s.quantile(.975)),log_loss_median=('log_loss','median'),ece_median=('ece','median'),auc_median=('auc','median')).reset_index().to_csv(OUT/'loco_local_calibration_summary.csv',index=False); pd.DataFrame(prov).to_csv(OUT/'loco_provenance.csv',index=False)
 # Competition-aware unseen-team comparison.
 tm=[]; tpred=[]; eligible=[(t,g) for t,g in x.groupby('team_id') if len(g)>=250]
 for tid,g in eligible:
  train=x[x.team_id!=tid]; ma=model(); ma.fit(train[NUM+CAT],train.R1.astype(int)); mb=model(CAT+['cohort']); mb.fit(train[NUM+CAT+['cohort']],train.R1.astype(int)); pa=ma.predict_proba(g[NUM+CAT])[:,1]; pb=mb.predict_proba(g[NUM+CAT+['cohort']])[:,1]
  A=metrics(g.R1.astype(int).to_numpy(),pa); B=metrics(g.R1.astype(int).to_numpy(),pb); tm.append({'held_out_team_id':tid,'held_out_team_name':g.team.iloc[0],'AUC_A':A['auc'],'AUC_B':B['auc'],'delta_AUC':B['auc']-A['auc'],'Brier_A':A['brier'],'Brier_B':B['brier'],'delta_Brier':B['brier']-A['brier'],'log_loss_A':A['log_loss'],'log_loss_B':B['log_loss'],'ECE_A':A['ece'],'ECE_B':B['ece'],'gap_A':A['gap'],'gap_B':B['gap']}); tpred.append(pd.DataFrame({'event_id':g.event_id,'team_id':tid,'cohort':g.cohort,'p_A':pa,'p_B':pb,'p_frozen':g.rho_TF}))
 tmf=pd.DataFrame(tm); tmf.to_csv(OUT/'unseen_team_competition_metrics.csv',index=False); pd.concat(tpred).to_parquet(OUT/'unseen_team_competition_predictions.parquet',index=False); summary={'n_teams':len(tmf)}
 for c in ['delta_AUC','delta_Brier']:
  summary[c+'_median']=float(tmf[c].median()); summary[c+'_iqr']=float(tmf[c].quantile(.75)-tmf[c].quantile(.25)); summary[c+'_p10']=float(tmf[c].quantile(.1)); summary[c+'_p90']=float(tmf[c].quantile(.9))
 pd.DataFrame([summary]).to_csv(OUT/'unseen_team_competition_summary.csv',index=False)
 # xT regimes on fixed five-match local calibration split and evaluation actions.
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet').merge(base,on='event_id',how='inner'); rows=[]; ranks=[]; act=[]
 for cohort,g in v.groupby('cohort',sort=True):
  mids=sorted(g.match_id.unique()); caids=mids[:min(5,len(mids)-1)]; ca=g[g.match_id.isin(caids)]; ev=g[~g.match_id.isin(caids)].copy(); xr=x[x.cohort==cohort].set_index('event_id'); ev=ev[ev.event_id.isin(xr.index)].copy(); ev['p_loco']=ev.event_id.map(xr.p_loco); a=intercept_only(ca.R1.to_numpy(),ca.event_id.map(xr.p_loco).dropna().to_numpy()) if len(ca)>0 else 0; aa,bb=calibration(ca.R1.to_numpy(),ca.event_id.map(xr.p_loco).dropna().to_numpy()) if len(ca)>1 else (a,1); ev['R0']=ev.rho_TF; ev['R1']=ev.p_loco; ev['R2']=expit(logit(np.clip(ev.p_loco,1e-5,1-1e-5))+a); ev['R3']=expit(aa+bb*logit(np.clip(ev.p_loco,1e-5,1-1e-5))); success=ev.success_payoff_TF-ev.origin_value_TF; failure=ev.failure_cost_TF
  for r in ['R0','R1','R2','R3']: ev['xt_'+r]=ev[r]*success-(1-ev[r])*failure
  for u,w in [('R0','R1'),('R1','R2'),('R1','R3')]: rows.append({'cohort':cohort,'comparison':u+'_vs_'+w,'pearson':ev['xt_'+u].corr(ev['xt_'+w]),'spearman':ev['xt_'+u].corr(ev['xt_'+w],method='spearman'),'mean_abs_value_change':(ev['xt_'+u]-ev['xt_'+w]).abs().mean(),'evaluation_n':len(ev)})
  for key in ['R0','R1','R2','R3']:
   q=ev.groupby(['keeper_id','keeper']).agg(n=('xt_'+key,'size'),value=('xt_'+key,'mean')).query('n>=25').reset_index(); q['rank']=q.value.rank(ascending=False,method='average'); q['regime']=key; q['cohort']=cohort; ranks.append(q)
  act.append(ev[['event_id','cohort','keeper_id','keeper','match_id','R0','R1','R2','R3','xt_R0','xt_R1','xt_R2','xt_R3']])
 pd.concat(act).to_parquet(OUT/'xtgk_patch_action_values.parquet',index=False); pd.DataFrame(rows).to_csv(OUT/'xtgk_patch_propagation.csv',index=False); rk=pd.concat(ranks); rk.to_csv(OUT/'xtgk_patch_keeper_ranks.csv',index=False)
 # within-cohort shifts relative to R1
 out=[]
 for (c,kid,name),q in rk.groupby(['cohort','keeper_id','keeper']):
  b=q[q.regime=='R1'].iloc[0]
  for regime in ['R0','R2','R3']:
   z=q[q.regime==regime].iloc[0]; out.append({'cohort':c,'keeper_id':kid,'keeper':name,'regime':regime,'n':int(z.n),'rank_shift':abs(z['rank']-b['rank'])})
 pd.DataFrame(out).to_csv(OUT/'xtgk_patch_keeper_rank_shifts.csv',index=False)
 print('R3.2A Patch 1 outputs written')
if __name__=='__main__': main()
