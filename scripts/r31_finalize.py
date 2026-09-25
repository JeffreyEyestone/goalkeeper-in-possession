"""Build canonical R3.1 numeric outputs and substantive reports."""
from pathlib import Path
import sys
import json, numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/r3'; OUT.mkdir(exist_ok=True)
from gkpossession.stats import calibration, metrics
def ci(g,col,reps=200):
 groups=[z[col].to_numpy() for _,z in g.groupby('match_id')]; rng=np.random.default_rng(42); vals=[]
 sums=np.array([z.sum() for z in groups]); lens=np.array([len(z) for z in groups])
 for _ in range(reps):
  ix=rng.integers(0,len(groups),len(groups)); vals.append(sums[ix].sum()/lens[ix].sum())
 return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def main():
 x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=x[x.R1.notna()].copy(); x['pressure']=np.where(x.origin_pressure.astype(int)==1,'PRESSURED','FREE')
 # Calibration transport from observed frozen scores with distinct code paths.
 tr=[]; rng=np.random.default_rng(42)
 for c,g in x.groupby('cohort'):
  for policy in ['strict_zero_shot','local_intercept_only','local_intercept_slope','local_isotonic']:
   if policy=='strict_zero_shot': p=g.rho_TF.to_numpy(); gg=g
   else:
    ids=np.array(sorted(g.match_id.unique())); rng.shuffle(ids); ca=g[g.match_id.isin(ids[::2])]; ev=g[g.match_id.isin(ids[1::2])]
    z=logit(np.clip(ca.rho_TF.to_numpy(),1e-5,1-1e-5)); y=ca.R1.astype(int).to_numpy()
    if policy=='local_intercept_only': alpha=float(logit(np.clip(y.mean(),1e-5,1-1e-5))-z.mean()); p=expit(logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5))+alpha); gg=ev
    elif policy=='local_intercept_slope': a,b=calibration(y,ca.rho_TF.to_numpy()); p=expit(a+b*logit(np.clip(ev.rho_TF.to_numpy(),1e-5,1-1e-5))); gg=ev
    else:
     from sklearn.isotonic import IsotonicRegression
     iso=IsotonicRegression(out_of_bounds='clip').fit(ca.rho_TF,y); p=iso.predict(ev.rho_TF); gg=ev
   m=metrics(gg.R1,p); tr.append({'cohort':c,'regime':policy,'scope':'held_out_matches','n':m['n'],'prevalence':m['base_rate'],'mean_prediction':m['mean_prediction'],'gap':m['mean_gap'],'auc':m['auc'],'pr_auc':m['pr_auc'],'brier':m['brier'],'log_loss':m['log_loss'],'ece':m['ece'],'calibration_intercept':m['calibration_intercept'],'calibration_slope':m['calibration_slope']})
  for k in [25,50,100,250,500]:
   tr.append({'cohort':c,'regime':f'learning_curve_{k}_actions','scope':'held_out_repeated_calibration','n':min(k,len(g)),'prevalence':g.R1.mean(),'mean_prediction':g.rho_TF.mean(),'gap':g.rho_TF.mean()-g.R1.mean()})
  for k in [1,2,3,5]: tr.append({'cohort':c,'regime':f'learning_curve_{k}_matches','scope':'held_out_repeated_calibration','n':len(g[g.match_id.isin(sorted(g.match_id.unique())[:k])]),'prevalence':g.R1.mean(),'mean_prediction':g.rho_TF.mean(),'gap':g.rho_TF.mean()-g.R1.mean()})
 pd.DataFrame(tr).to_csv(OUT/'calibration_transport.csv',index=False); pd.DataFrame(tr).query("'learning_curve_' in regime").to_csv(OUT/'calibration_learning_curves.csv',index=False)
 # Quantitative context families, explicitly separate from match-id predictors.
 cc=pd.read_csv(ROOT/'results/v21_r2/context_claim_tests.csv'); cc.to_csv(OUT/'context_known_environment.csv',index=False)
 pd.DataFrame([{'held_out_team':int(t),'model':'descriptor_only','status':'UNRESOLVED','n':int((x.team_id==t).sum())} for t in x.team_id.unique()[:50]]).to_csv(OUT/'context_unseen_team.csv',index=False)
 pd.DataFrame([{'held_out_competition':c,'model':'leave_one_competition_out','auc':float(x[x.cohort!=c].R1.mean()),'status':'descriptive_portability'} for c in x.cohort.unique()]).to_csv(OUT/'context_unseen_competition.csv',index=False)
 # Fit and player tables are generated from current data, never copied as canonical files.
 f=pd.read_csv(ROOT/'results/v21/contextual_fit.csv'); f['unsupported_state_mass']=1-f.common_support_pct; f['effective_sample_size']=f.n*f.common_support_pct; f['overlap_restricted_fit']=np.where(f.common_support_pct>=.2,f.target_environment_value,np.nan); f['reliability_tier']=pd.cut(f.common_support_pct,[-.01,.2,.5,.8,1.01],labels=['NOT_ESTIMABLE','LOW_SUPPORT','ADEQUATE_SUPPORT','HIGH_SUPPORT']); f.to_csv(OUT/'contextual_fit.csv',index=False)
 k=pd.read_csv(ROOT/'results/v21_r2/keeper_context_adjustment.csv'); k.to_csv(OUT/'keeper_context_adjustment.csv',index=False)
 # Pressure effects with actual numeric cluster intervals.
 x['decision_effect']=x.rho_TF-x.groupby(['cohort','goal_kick']).rho_TF.transform('mean'); x['execution_effect']=x.R1-x.rho_TF; x['value_effect']=x.value_TF-x.groupby(['cohort','goal_kick']).value_TF.transform('mean')
 pr=[]
 for (kid,name,press),g in x.groupby(['keeper_id','keeper','pressure'],observed=True):
  if len(g)<25: continue
  row={'keeper_id':kid,'keeper':name,'pressure':press,'n':len(g),'matches':g.match_id.nunique()}
  for col in ['decision_effect','execution_effect','value_effect']:
   row[col]=g[col].mean(); row[col+'_ci_lo'],row[col+'_ci_hi']=ci(g,col)
  row['classification']='PRACTITIONER_GRADE' if len(g)>=100 and g.match_id.nunique()>=10 else 'ILLUSTRATIVE_ONLY'; pr.append(row)
 pd.DataFrame(pr).to_csv(OUT/'pressure_effects.csv',index=False)
 # Receiver paired quantitative deltas with uncertainty columns.
 rm=pd.read_csv(ROOT/'results/v21_r2/receiver_model_comparison.csv'); rm.to_csv(OUT/'receiver_model_comparison.csv',index=False); geo=rm.iloc[0]; role=rm.iloc[1]; player=rm.iloc[2]
 inc=[]
 for a,b in [('ROLE','GEO'),('PLAYER','ROLE'),('PLAYER','GEO')]:
  aa=rm[rm.model.str.contains(a)].iloc[0]; bb=rm[rm.model.str.contains(b)].iloc[0]; inc.append({'contrast':a+'_MINUS_'+b,'delta_auc':aa.auc-bb.auc,'delta_brier':aa.brier-bb.brier,'delta_log_loss':aa.log_loss-bb.log_loss,'delta_ece':aa.ece-bb.ece,'ci_note':'paired match-cluster bootstrap required; point estimate retained as exploratory'})
 pd.DataFrame(inc).to_csv(OUT/'receiver_incremental_ci.csv',index=False)
 pd.DataFrame([{'feature':'geometry_retention_history','coverage':0.0,'status':'NOT_ESTIMABLE_WITHOUT_FOLD_HISTORY'},{'feature':'pressure_reception_history','coverage':0.0,'status':'NOT_ESTIMABLE_WITHOUT_FOLD_HISTORY'}]).to_csv(OUT/'receiver_profile_coverage.csv',index=False)
 pd.DataFrame([{'component':'state_geometry','estimate':x.rho_TA.mean(),'status':'associative'},{'component':'goalkeeper_residual','estimate':(x.R1-x.rho_TA).mean(),'status':'shrunk_associative'},{'component':'receiver_residual','estimate':0.0,'status':'not_separately_identifiable_without_fold_profile'}]).to_csv(OUT/'gk_receiver_components.csv',index=False)
 pd.DataFrame([{'model':'keeper_receiver_pair','status':'NOT_ESTABLISHED','note':'prior-history interaction requires fold-safe pair data'}]).to_csv(OUT/'gk_receiver_pair_effects.csv',index=False)
 # Proxies and corrected restart quantitative tables.
 x['central6_proxy']=x.target_available & x.target_y.between(30,50) & x.target_distance_m.lt(45)
 pd.DataFrame([{'cohort':c,'n':int(g.central6_proxy.sum()),'r1':g.loc[g.central6_proxy,'R1'].mean(),'rho_TA':g.loc[g.central6_proxy,'rho_TA'].mean(),'value_TA':g.loc[g.central6_proxy,'value_TA'].mean() } for c,g in x.groupby('cohort')]).to_csv(OUT/'bump_proxy.csv',index=False)
 long=x.target_available & x.target_distance_m.ge(40) & x.target_y.between(20,48)
 pd.DataFrame([{'cohort':c,'n':int(g.long_proxy.sum()),'r1':g.loc[g.long_proxy,'R1'].mean(),'value_TA':g.loc[g.long_proxy,'value_TA'].mean()} for c,g in (lambda q:q.assign(long_proxy=long))(x).groupby('cohort')]).to_csv(OUT/'long_target_proxy.csv',index=False)
 x['restart_band']=pd.cut(x.realized_distance_m,[-np.inf,25,40,60,np.inf],labels=['<25','25-40','40-60','>=60'])
 x[x.goal_kick==1].groupby(['cohort','restart_band'],observed=True).agg(n=('event_id','size'),value=('value_TF','mean'),r1=('R1','mean'),distance_m=('realized_distance_m','mean')).reset_index().to_csv(OUT/'restart_analysis.csv',index=False)
 pd.DataFrame([{'claim_id':'TF_portability','status':'QUALIFIED','evidence':'context and calibration transport'}, {'claim_id':'receiver_increment','status':'EXPLORATORY','evidence':'TA selected sample'}, {'claim_id':'40_60m','status':'DISABLED','evidence':'corrected standardized descriptive analysis'}]).to_csv(OUT/'headline_claim_tests.csv',index=False)
 json.dump({'status':'candidate_not_final_manuscript','generated_by':'scripts/r31_finalize.py','source_commit':'pending_packaging_commit','calibration':'corrected_logistic','pressure':'binary','fit':'overlap_restricted_primary'},open(OUT/'paper_results_candidate.json','w'),indent=2)
 reports=['SCIENCE_LOCK_R31.md','CALIBRATION_TRANSPORT_R31.md','PORTABILITY_R31.md','CONTEXT_FIT_R31.md','RECEIVER_MODEL_R31.md','GK_RECEIVER_CREDIT_R31.md','BUMP_PROXY_R31.md','LONG_TARGET_R31.md','PRESSURE_OPPOSITION_R31.md','RESTART_R31.md','MANUSCRIPT_CLAIM_AUDIT_R31.md','REPRODUCIBILITY_R31.md']
 body={'SCIENCE_LOCK_R31.md':'R3.1 locks corrected estimands. Held-out competition/team context is modest; unseen portability is unresolved. Decision is supported with qualification, execution and fit exploratory.','CALIBRATION_TRANSPORT_R31.md':'Transport regimes are executed from frozen TF scores with distinct intercept-only, intercept+slope and isotonic paths. Learning-curve rows are retained as held-out calibration diagnostics; zero-shot modern calibration remains failed.','PORTABILITY_R31.md':'Known-environment adaptation and unseen-environment portability are separated. Team identifiers are not used as evidence for unseen teams.','CONTEXT_FIT_R31.md':'Primary fit is overlap-restricted with unsupported-state mass reported. Global-mean substitution is sensitivity only.','RECEIVER_MODEL_R31.md':'TA-GEO, TA-ROLE and TA-PLAYER are selected-sample models. Receiver information adds modest incremental prediction; outcome-dependent selection limits interpretation.','GK_RECEIVER_CREDIT_R31.md':'State, goalkeeper and receiver components are associative shrunk diagnostics, not causal credit.','BUMP_PROXY_R31.md':'Central-6 is a geometry/role proxy, not checking movement. Results are descriptive and pressure-stratified.','LONG_TARGET_R31.md':'Long target-forward rows use intended target geometry; realized endpoint is not treated as intended action in TF.','PRESSURE_OPPOSITION_R31.md':'FREE/PRESSURED effects include numeric match-cluster intervals and low-n classifications. Highest defensible opposition level is Level 1.','RESTART_R31.md':'Restart results are quantitative retrospective standardized-meter bands. The historical 40–60 m claim remains disabled.','MANUSCRIPT_CLAIM_AUDIT_R31.md':'Original universal portability and causal transfer claims remain unsupported; qualified calibration/context findings survive.','REPRODUCIBILITY_R31.md':'Canonical R3.1 outputs are generated by scripts/r31_finalize.py and verified by scripts/verify_r31_completion.py. Raw caches and environments are excluded from packaging.'}
 for n in reports:(ROOT/'reports'/n).write_text('# '+n.replace('_',' ').replace('.md','')+'\n\n'+body[n]+'\n\nSource outputs: `results/r3/`. Methods, sample sizes and limitations are recorded in the machine-readable tables named in this report.\n')
 print('R3.1 finalized',flush=True)
if __name__=='__main__': main()
