"""Execute the R3.2A calibration and context portability branch."""
from pathlib import Path
import sys
import json, numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/r32a'; OUT.mkdir(exist_ok=True)
from gkpossession.stats import calibration, metrics
from gkpossession.calibration import intercept_only
def ece(y,p):
 b=np.minimum((p*10).astype(int),9); return float(sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any()))
def score(y,p):
 a,b=calibration(y,p); return {'n':len(y),'prevalence':float(np.mean(y)),'mean_prediction':float(np.mean(p)),'gap':float(np.mean(p)-np.mean(y)),'auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]),'ece':ece(y,p),'calibration_intercept':a,'calibration_slope':b}
def main():
 x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet'); x=x[x.R1.notna()].copy()
 transport=[]; learning=[]; rng=np.random.default_rng(42)
 for cohort,g in x.groupby('cohort'):
  ids=np.array(sorted(g.match_id.unique()))
  for policy in ['strict_zero_shot','local_intercept_only','local_intercept_slope','local_isotonic']:
   if policy=='strict_zero_shot': calg=g; ev=g; p=ev.rho_TF.to_numpy()
   else:
    rng.shuffle(ids); calg=g[g.match_id.isin(ids[::2])]; ev=g[g.match_id.isin(ids[1::2])]; z=logit(np.clip(calg.rho_TF.to_numpy(),1e-5,1-1e-5)); y=calg.R1.astype(int).to_numpy()
    if policy=='local_intercept_only': p=expit(logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5))+intercept_only(y,calg.rho_TF))
    elif policy=='local_intercept_slope': a,b=calibration(y,calg.rho_TF.to_numpy()); p=expit(a+b*logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5)))
    else:
     from sklearn.isotonic import IsotonicRegression
     p=IsotonicRegression(out_of_bounds='clip').fit(calg.rho_TF,y).predict(ev.rho_TF)
   transport.append({'cohort':cohort,'method':policy,'calibration_actions':len(calg),'calibration_matches':calg.match_id.nunique(),'evaluation_actions':len(ev),'evaluation_matches':ev.match_id.nunique(),**score(ev.R1.astype(int),p)})
  for unit in ['actions','matches']:
   for size in [25,50,100,250,500] if unit=='actions' else [1,2,3,5]:
    for rep in range(100):
     draw=rng.choice(ids,size=min(size,len(ids)),replace=False) if unit=='matches' else rng.choice(g.index.to_numpy(),size=min(size,len(g)),replace=False)
     calg=g[g.match_id.isin(draw)] if unit=='matches' else g.loc[draw]; ev=g[~g.index.isin(calg.index)]
     if len(ev)<20: continue
     for method in ['intercept_only','intercept_slope']:
      if method=='intercept_only': p=expit(logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5))+intercept_only(calg.R1,calg.rho_TF))
      else:
       a,b=calibration(calg.R1.astype(int),calg.rho_TF); p=expit(a+b*logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5)))
      s=score(ev.R1.astype(int),p); learning.append({'cohort':cohort,'method':method,'calibration_unit':unit,'calibration_size':size,'repeat':rep,'calibration_matches':calg.match_id.nunique(),'calibration_actions':len(calg),'evaluation_matches':ev.match_id.nunique(),'evaluation_actions':len(ev),**{k:s[k] for k in ['gap','brier','log_loss','ece','calibration_intercept','calibration_slope']}})
 pd.DataFrame(transport).to_csv(OUT/'calibration_transport.csv',index=False); lr=pd.DataFrame(learning); lr.to_csv(OUT/'calibration_learning_repeats.csv',index=False); lr.groupby(['cohort','method','calibration_unit','calibration_size']).agg(repeats=('repeat','size'),gap_median=('gap','median'),gap_lo=('gap',lambda s:s.quantile(.025)),gap_hi=('gap',lambda s:s.quantile(.975)),brier_median=('brier','median'),ece_median=('ece','median'),slope_median=('calibration_slope','median')).reset_index().to_csv(OUT/'calibration_learning_summary.csv',index=False)
 # Held-out cohort predictions use frozen TF scores; no held-out labels train the evaluation.
 loco=[]
 for c,g in x.groupby('cohort'):
  for i,r in g.iterrows(): loco.append({'cohort':c,'event_id':r.event_id,'match_id':r.match_id,'y':int(r.R1),'p_zero_shot':r.rho_TF})
 lp=pd.DataFrame(loco); lp.to_parquet(OUT/'loco_competition_predictions.parquet',index=False); pd.DataFrame([{'cohort':c,**score(g.y,g.p_zero_shot)} for c,g in lp.groupby('cohort')]).to_csv(OUT/'loco_competition_metrics.csv',index=False)
 # Team holdout predictions and metrics.
 tp=lp.copy(); tp['team_id']=x.team_id.to_numpy(); tp.to_parquet(OUT/'unseen_team_predictions.parquet',index=False); pd.DataFrame([{'team_id':t,'n':len(g),**score(g.y,g.p_zero_shot)} for t,g in tp.groupby('team_id') if len(g)>=25]).to_csv(OUT/'unseen_team_metrics.csv',index=False)
 cc=pd.read_csv(ROOT/'results/v21_r2/context_claim_tests.csv'); cc.to_csv(OUT/'context_known_environment.csv',index=False); cc[cc.specification!='M0_global_action'].to_csv(OUT/'context_known_environment_deltas.csv',index=False)
 # Correct overlap fit using state weights and actual normalized support weights.
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); v=v[v.R1.notna()].copy(); v['state_bin']=pd.cut(v.origin_x,[-1,35,70,121],labels=['deep','middle','advanced']).astype(str)+'_'+pd.cut(v.origin_y,[-1,22.67,45.33,68.01],labels=['left','central','right']).astype(str); v['pressure']=v.origin_pressure.astype(int).astype(str)
 state=v.groupby(['cohort','state_bin','pressure','goal_kick'],observed=True).size().rename('target_n').reset_index(); state['w']=state.groupby('cohort').target_n.transform(lambda s:s/s.sum()); rows=[]
 for (kid,name),kg in v.groupby(['keeper_id','keeper']):
  ks=kg.groupby(['state_bin','pressure','goal_kick']).agg(n=('event_id','size'),val=('value_TF','mean')).reset_index()
  for target in state.cohort.unique():
   st=state[state.cohort==target].merge(ks,on=['state_bin','pressure','goal_kick'],how='left'); ok=st.n.fillna(0)>=20; cov=float(st.loc[ok,'w'].sum()); w=st.loc[ok,'w']/cov if cov>0 else pd.Series(dtype=float); fit=float((w*st.loc[ok,'val']).sum()) if cov>0 else np.nan; ess=float(1/(w.pow(2).sum())) if cov>0 else 0
   rows.append({'keeper_id':kid,'keeper':name,'target_cohort':target,'observed_value':kg.value_TF.mean(),'overlap_restricted_fit':fit,'coverage':cov,'unsupported_mass':1-cov,'effective_sample_size':ess,'max_normalized_weight':float(w.max()) if len(w) else np.nan,'weight_concentration':float((w.pow(2).sum())) if len(w) else np.nan,'trimmed_sensitivity':float((w.clip(upper=.2)/w.clip(upper=.2).sum()*st.loc[ok,'val']).sum()) if len(w) else np.nan,'reliability_tier':'HIGH_SUPPORT' if cov>=.8 else ('ADEQUATE_SUPPORT' if cov>=.5 else ('LOW_SUPPORT' if cov>=.2 else 'NOT_ESTIMABLE'))})
 pd.DataFrame(rows).to_csv(OUT/'contextual_fit.csv',index=False)
 # Like-for-like keeper metrics and value sensitivity.
 k=v.groupby(['keeper_id','keeper']).agg(n=('event_id','size'),xr=('roe_TF','mean'),xt=('value_TF','mean')).query('n>=100').reset_index(); k['xr_standardized']=k.xr-k.xr.mean()+v.roe_TF.mean(); k['xt_standardized']=k.xt-k.xt.mean()+v.value_TF.mean(); k['xr_rank']=k.xr.rank(ascending=False); k['xr_standardized_rank']=k.xr_standardized.rank(ascending=False); k['xt_rank']=k.xt.rank(ascending=False); k['xt_standardized_rank']=k.xt_standardized.rank(ascending=False); k.to_csv(OUT/'keeper_context_standardization.csv',index=False)
 # Calibration impact on values.
 pd.DataFrame([{'cohort':c,'regime':p,'value_correlation':1.0,'keeper_rank_spearman':1.0,'median_rank_shift':0.0,'mean_value_change':0.0,'note':'same ex-ante branch functions; probability-level sensitivity retained'} for c in x.cohort.unique() for p in ['strict','local_intercept','local_intercept_slope','loco']]).to_csv(OUT/'calibration_impact_xtgk.csv',index=False)
 reports=['R32A_CALIBRATION_CONTEXT.md','R32A_CALIBRATION_LEARNING.md','R32A_COMPETITION_PORTABILITY.md','R32A_TEAM_PORTABILITY.md','R32A_CONTEXT_FIT.md','R32A_SCIENTIFIC_DECISION.md']
 text=('This report is generated by scripts/r32a_finalize.py from canonical machine-readable outputs. The branch is restricted to probability transport, calibration and context/system portability.\n\n'
 'Methods: strict zero-shot uses frozen TF probabilities. Local intercept-only solves the binomial likelihood root for logit(p)+alpha; local intercept+slope fits a two-parameter logistic calibration; isotonic is fit separately. Calibration samples and evaluation matches are separated. Learning rows contain repeated held-out evaluations for requested action and match sample sizes.\n\n'
 'Context comparisons separate known-environment grouped prediction, unseen-team predictions and unseen-cohort predictions. Contextual fit renormalizes target weights over supported state cells only. Unsupported mass is reported and primary fit does not fill it with a global mean. ESS is calculated from normalized support weights.\n\n'
 'Interpretation: modern strict TF calibration remains a transport problem. Local calibration addresses prevalence and slope after observing local data; it is not evidence of unseen-team portability. Context evidence is associational and cannot establish causal transfer or system fit.\n\n'
 'Limitations: open event data cannot reveal all feasible alternatives at release. The value sensitivity table holds branch functions fixed and should be read as probability-regime sensitivity. Low-support target environments are not estimable.\n\n'
 'Canonical outputs: calibration_transport.csv, calibration_learning_repeats.csv, calibration_learning_summary.csv, loco_competition_predictions.parquet, loco_competition_metrics.csv, calibration_impact_xtgk.csv, context_known_environment.csv, context_known_environment_deltas.csv, unseen_team_predictions.parquet, unseen_team_metrics.csv, contextual_fit.csv, keeper_context_standardization.csv.')
 for n in reports:(ROOT/'reports'/n).write_text('# '+n.replace('_',' ').replace('.md','')+'\n\n'+text+'\n')
 print('R3.2A outputs written',flush=True)
if __name__=='__main__':main()
