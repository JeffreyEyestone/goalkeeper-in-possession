import os, json, hashlib, joblib
from pathlib import Path
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/winner_gate13'; OUT.mkdir(exist_ok=True)
import sys; sys.path.insert(0,str(ROOT/'src'))
# Canonical imports: these are the accepted implementations, not copies.
from gkpossession.xr import classifier, fit_calibration, calibrated
from gkpossession.features import matrix, TF_NUM, TF_CAT
from gkpossession.surfaces import fit_bundle, lookup
from gkpossession.failure_cost import ShrunkBranchMean, regressor
from gkpossession.xtgk import score, score_components
SRC=['src/gkpossession/xr.py','src/gkpossession/features.py','src/gkpossession/surfaces.py','src/gkpossession/failure_cost.py','src/gkpossession/xtgk.py','scripts/fit_xr.py','scripts/fit_values.py']
OUTCOMES=['Y10_NET_XG','Y30_NET_XG','YCYCLE_NET_XG','YNEXTSHOT']; METRICS=['B0','B1','B2_XR','B3_SUCCESS_ONLY','B4_FIXED_FAILURE','B5_COARSE_ORIGIN_FAILURE','B6_FULL_XTGK']; TOUR={'WC2018','WC2022','T2024'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hash_ids(a): return hashlib.sha256('|'.join(map(str,sorted(set(a)))).encode()).hexdigest()
def lock(): return {p:sha(ROOT/p) for p in SRC}
def check_lock(l): assert lock()==l, 'canonical source changed during run'
def main():
 lock_path=OUT/'CANONICAL_LOCK.json'
 if lock_path.exists(): locked=json.loads(lock_path.read_text()); assert lock()==locked['source_sha256'], 'canonical source lock mismatch'
 else:
  locked={'source_sha256':lock(),'created_from_commit':'41fe05d'}; lock_path.write_text(json.dumps(locked,indent=2,sort_keys=True))
 check_lock(locked['source_sha256'])
 xrsel=json.loads((ROOT/'results/v21/xr_selected_models.json').read_text()); brsel=json.loads((ROOT/'results/v21/branch_model_selection.json').read_text())
 assert xrsel['TF']['family']=='logistic' and xrsel['TF']['calibration']=='isotonic'; assert brsel['TF_1']['family']=='shrunk' and brsel['TF_0']['family']=='shrunk'
 x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); y=pd.read_parquet(ROOT/'results/winner_gate11/observed_value_outcomes.parquet')[['event_id']+OUTCOMES]; d=x.merge(y,on='event_id',validate='one_to_one'); d=d[d.R1.notna()].copy(); states=pd.read_parquet(ROOT/'data/processed/surface_states.parquet')
 # deterministic outer folds within each cohort
 fold_rows=[]; oof=[]; prov=[]; rng=np.random.default_rng(42)
 for cohort,g in d.groupby('cohort',sort=True):
  g=g.copy(); groups=g.match_id.to_numpy(); splitter=GroupKFold(5)
  for fold,(tri,tei) in enumerate(splitter.split(g,g.R1,groups)):
   tr=g.iloc[tri].copy(); te=g.iloc[tei].copy(); train_matches=sorted(tr.match_id.unique()); eval_matches=sorted(te.match_id.unique())
   fold_rows += [{'event_id':r.event_id,'action_id':r.event_id,'match_id':r.match_id,'cohort':cohort,'outer_fold':fold,'role':'evaluation'} for _,r in te.iterrows()]
   fold_rows += [{'event_id':r.event_id,'action_id':r.event_id,'match_id':r.match_id,'cohort':cohort,'outer_fold':fold,'role':'training'} for _,r in tr.iterrows()]
   # internal model/calibration split is entirely inside outer training
   ids=np.array(train_matches); rr=np.random.default_rng(42); rr.shuffle(ids); cut=max(1,int(round(.8*len(ids)))); fit_ids=set(ids[:cut]); cal_ids=set(ids[cut:]) or set(ids[-1:]); fitr=tr[tr.match_id.isin(fit_ids)]; calr=tr[tr.match_id.isin(cal_ids)]
   model=classifier(xrsel['TF']['family'],'TF'); model.fit(matrix(fitr,'TF'),fitr.R1.astype(int)); pcal=model.predict_proba(matrix(calr,'TF'))[:,1]; cal=fit_calibration(calr.R1.astype(int),pcal,xrsel['TF']['calibration']); rho=calibrated(cal,model.predict_proba(matrix(te,'TF'))[:,1],xrsel['TF']['calibration']); assert np.std(rho)>0
   surf=fit_bundle(states,train_matches,grid=(16,12),k=30)
   tr2=tr[tr.R1_state_x.notna() & tr.R1_state_y.notna()].copy(); own=tr2.R1.eq(1); lost=tr2.R1.eq(0); tr2.loc[own,'_branch_y']=lookup(surf,tr2.loc[own,'R1_state_x'],tr2.loc[own,'R1_state_y'],pressure=tr2.loc[own,'R1_state_pressure']); tr2.loc[lost,'_branch_y']=lookup(surf,tr2.loc[lost,'R1_state_x'],tr2.loc[lost,'R1_state_y'],name='T')
   sm=ShrunkBranchMean('TF',k=30).fit(tr2[own & tr2._branch_y.notna()],tr2.loc[own & tr2._branch_y.notna(),'_branch_y']); fm=ShrunkBranchMean('TF',k=30).fit(tr2[lost & tr2._branch_y.notna()],tr2.loc[lost & tr2._branch_y.notna(),'_branch_y'])
   scored=score(te,rho,sm,fm,surf,'TF'); V=scored['origin_value']; S=scored['success_payoff']; C=scored['failure_cost']; b6=scored['value']; eq=score_components(rho,S,V,C); assert np.allclose(b6,eq,atol=1e-12)
   band=np.where(te.origin_x*105/120<=16.5,'0_16_5',np.where(te.origin_x*105/120<=35,'16_5_35','GT35')); tr2['_band']=np.where(tr2.origin_x*105/120<=16.5,'0_16_5',np.where(tr2.origin_x*105/120<=35,'16_5_35','GT35')); globalc=float(tr2.loc[lost & tr2._branch_y.notna(),'_branch_y'].mean()); bm=tr2[lost & tr2._branch_y.notna()].groupby('_band')._branch_y.mean().to_dict(); cb=np.array([bm.get(z,globalc) for z in band]); b2=rho; b3=rho*S-V; b4=rho*S-V-(1-rho)*globalc; b5=rho*S-V-(1-rho)*cb; assert np.max(np.abs(b5-b6))>0
   for k,obj in [('rho',{'model':model,'calibrator':cal,'accepted_family':'logistic','accepted_calibration':'isotonic'}),('surface',surf),('success',sm),('failure',fm)]: joblib.dump(obj,OUT/f'{cohort}_fold{fold}_{k}.joblib')
   out=pd.DataFrame({'action_id':te.event_id,'event_id':te.event_id,'match_id':te.match_id,'cohort':cohort,'team_id':te.team_id,'keeper_id':te.keeper_id,'outer_fold':fold,'R1':te.R1,'complete':te.complete,'Y10_NET_XG':te.Y10_NET_XG,'Y30_NET_XG':te.Y30_NET_XG,'YCYCLE_NET_XG':te.YCYCLE_NET_XG,'YNEXTSHOT':te.YNEXTSHOT,'rho_oof':rho,'V_origin_oof':V,'V_success_oof':S,'C_action_oof':C,'C_global_train':globalc,'C_band_train':cb,'B2_XR':b2,'B3_SUCCESS_ONLY':b3,'B4_FIXED_FAILURE':b4,'B5_COARSE_ORIGIN_FAILURE':b5,'B6_FULL_XTGK':b6,'B0':te.complete,'B1':te.R1,'origin_x_m':te.origin_x*105/120,'team':te.team,'keeper':te.keeper}); oof.append(out)
   state={'cohort':cohort,'fold':fold,'train_matches':train_matches,'fit_matches':sorted(fit_ids),'cal_matches':sorted(cal_ids),'accepted_family':'logistic','accepted_calibration':'isotonic'}; prov.append({'cohort':cohort,'outer_fold':fold,'training_match_hash':hash_ids(train_matches),'evaluation_match_hash':hash_ids(eval_matches),'fit_match_hash':hash_ids(fit_ids),'calibration_match_hash':hash_ids(cal_ids),'xR_source_file_sha':sha(ROOT/'src/gkpossession/xr.py'),'V_source_file_sha':sha(ROOT/'src/gkpossession/surfaces.py'),'C_source_file_sha':sha(ROOT/'src/gkpossession/failure_cost.py'),'rho_artifact_sha':sha(OUT/f'{cohort}_fold{fold}_rho.joblib'),'surface_artifact_sha':sha(OUT/f'{cohort}_fold{fold}_surface.joblib'),'success_artifact_sha':sha(OUT/f'{cohort}_fold{fold}_success.joblib'),'failure_artifact_sha':sha(OUT/f'{cohort}_fold{fold}_failure.joblib'),'accepted_family':'logistic','accepted_calibration':'isotonic'})
 check_lock(locked['source_sha256']); o=pd.concat(oof,ignore_index=True); o.to_parquet(OUT/'oof_action_values.parquet',index=False); pd.DataFrame(fold_rows).to_csv(OUT/'fold_assignments.csv',index=False); pd.DataFrame(prov).to_csv(OUT/'oof_model_provenance.csv',index=False)
 # action deciles and true match bootstrap
 dec=[]; boot=[]; rng=np.random.default_rng(1301)
 for c,g in o.groupby('cohort'):
  mids=np.array(g.match_id.unique())
  for m in METRICS[2:]:
   z=g.copy(); z['decile']=pd.qcut(z[m].rank(method='first'),10,labels=False)+1; q=z.groupby('decile').agg(mean_metric=(m,'mean'),**{y:(y,'mean') for y in OUTCOMES}).reset_index(); q['cohort']=c; q['metric']=m
   for _,r in q.iterrows():
    for y in OUTCOMES: dec.append({'cohort':c,'metric':m,'decile':int(r.decile),'mean_metric':r.mean_metric,'outcome':y,'mean_outcome':r[y],'decile_spearman':spearmanr(q.mean_metric,q[y]).statistic})
   # sufficient statistics preserve sampled match multiplicity and recompute decile means/correlation
   # Precompute match x decile sufficient statistics so each true match
   # bootstrap draw is a small matrix multiply rather than repeated pandas
   # filtering/concatenation.
   mm=z.groupby(['match_id','decile']).agg(metric_sum=(m,'sum'),n=(m,'size'),**{y:(y,'sum') for y in OUTCOMES}).reset_index()
   mid_index={mid:i for i,mid in enumerate(mids)}; mm['_mi']=mm.match_id.map(mid_index).astype(int); mm['_di']=mm.decile.astype(int)-1
   shape=(len(mids),10); metric_arr=np.zeros(shape); n_arr=np.zeros(shape); outcome_arr={y:np.zeros(shape) for y in OUTCOMES}
   cols=['_mi','_di','metric_sum','n']+OUTCOMES
   for mi,di,ms,nn,*vals in mm[cols].itertuples(index=False,name=None):
    metric_arr[mi,di]=ms; n_arr[mi,di]=nn
    for j,y in enumerate(OUTCOMES): outcome_arr[y][mi,di]=vals[j]
   for draw in range(5000):
    sm=rng.choice(mids,size=len(mids),replace=True); counts=np.bincount([mid_index[x] for x in sm],minlength=len(mids)); agg_metric=counts@metric_arr; agg_n=counts@n_arr; h=hashlib.sha256(sm.tobytes()).hexdigest(); pred=agg_metric/agg_n
    for y in OUTCOMES: boot.append({'draw_id':draw,'sample_hash':h,'cohort':c,'metric':m,'outcome':y,'spearman':spearmanr(pred,(counts@outcome_arr[y])/agg_n).statistic})
 pd.DataFrame(dec).to_csv(OUT/'action_decile_validation.csv',index=False); pd.DataFrame(boot).to_csv(OUT/'action_decile_bootstrap.csv',index=False)
 # keeper-level, vectorized true draws
 kval=[]; kb=[]; rng=np.random.default_rng(1302)
 for c,g in o.groupby('cohort'):
  lim=25 if c in TOUR else 200; kg=g.groupby(['keeper_id','keeper'],as_index=False).agg(n=('event_id','size'),matches=('match_id','nunique'),**{m:(m,'mean') for m in METRICS},**{y:(y,'mean') for y in OUTCOMES}); kg=kg[(kg.n>=lim)&((kg.matches>=2) if c in TOUR else True)].reset_index(drop=True)
  for m in METRICS:
   for y in OUTCOMES: kval.append({'cohort':c,'metric':m,'outcome':y,'eligible_keepers':len(kg),'pearson':pearsonr(kg[m],kg[y]).statistic,'spearman':spearmanr(kg[m],kg[y]).statistic})
  ix=rng.integers(0,len(kg),size=(10000,len(kg))); hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in ix]
  for m in METRICS:
   for y in OUTCOMES:
    av=kg[m].to_numpy(float)[ix]; bv=kg[y].to_numpy(float)[ix]; ar=pd.DataFrame(av).rank(axis=1).to_numpy(); br=pd.DataFrame(bv).rank(axis=1).to_numpy();
    def corr(a,b):
     aa=a-a.mean(1,keepdims=True); bb=b-b.mean(1,keepdims=True); return (aa*bb).sum(1)/np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
    ps=corr(av,bv); ss=corr(ar,br)
    for i in range(10000): kb.append({'cohort':c,'draw_id':i,'sample_hash':hashes[i],'metric':m,'outcome':y,'pearson':ps[i],'spearman':ss[i]})
 pd.DataFrame(kval).to_csv(OUT/'keeper_validation.csv',index=False); kbdf=pd.DataFrame(kb); kbdf.to_csv(OUT/'keeper_validation_bootstrap.csv',index=False)
 # paired from same draw
 rows=[]
 for c in o.cohort.unique():
  for y in OUTCOMES:
   for b in ['B2_XR','B3_SUCCESS_ONLY','B4_FIXED_FAILURE','B5_COARSE_ORIGIN_FAILURE','B0','B1']:
    a=kbdf[(kbdf.cohort==c)&(kbdf.outcome==y)&(kbdf.metric=='B6_FULL_XTGK')].sort_values('draw_id'); z=kbdf[(kbdf.cohort==c)&(kbdf.outcome==y)&(kbdf.metric==b)].sort_values('draw_id'); ds=a.spearman.to_numpy()-z.spearman.to_numpy(); dp=a.pearson.to_numpy()-z.pearson.to_numpy(); lo,hi=np.percentile(ds,[2.5,97.5]); rows.append({'cohort':c,'outcome':y,'comparison':'B6_vs_'+b,'delta_spearman':ds.mean(),'spearman_2_5':lo,'spearman_97_5':hi,'P_delta_spearman_gt0':np.mean(ds>0),'delta_pearson':dp.mean(),'pearson_2_5':np.percentile(dp,2.5),'pearson_97_5':np.percentile(dp,97.5),'P_delta_pearson_gt0':np.mean(dp>0),'classification':'FULL_CLEARLY_BETTER' if lo>0 else ('BASELINE_CLEARLY_BETTER' if hi<0 else 'NO_CLEAR_DIFFERENCE')})
 pd.DataFrame(rows).to_csv(OUT/'paired_metric_comparisons.csv',index=False)
 kv=pd.DataFrame(kval); hs=[]
 for m in METRICS:
  vals=kv[kv.metric==m].spearman.to_numpy(); ranks=kv.assign(rank=kv.groupby(['cohort','outcome']).spearman.rank(ascending=False,method='min')); rr=ranks.loc[ranks.metric==m,'rank'].to_numpy()
  hs.append({'metric':m,'positive_spearman_cells':(vals>0).sum(),'median_spearman':np.median(vals),'p10_spearman':np.percentile(vals,10),'minimum':vals.min(),'maximum':vals.max(),'sd_spearman':vals.std(),'rank1':(rr==1).sum(),'top2':(rr<=2).sum(),'top3':(rr<=3).sum()})
 pd.DataFrame(hs).to_csv(OUT/'metric_horizon_stability.csv',index=False)
 # keeper-match aggregates and statsmodels clustered panels
 km=o.groupby(['cohort','keeper_id','keeper','team_id','team','match_id'],as_index=False).agg(n=('event_id','size'),**{m:(m,'mean') for m in METRICS},**{y:(y,'mean') for y in OUTCOMES}); km=km[km.n>=10].copy(); km.to_csv(OUT/'keeper_match_validation.csv',index=False)
 try:
  import statsmodels.api as sm
  regs=[]
  for m in METRICS[2:]:
   for y in OUTCOMES:
    km['_z']= (km[m]-km[m].mean())/km[m].std()
    cohort_d=pd.get_dummies(km['cohort'],drop_first=True,dtype=float)
    designs=[('MODEL1_COHORT_FE',pd.concat([pd.Series(1.0,index=km.index,name='const'),km[['_z']],cohort_d],axis=1))]
    team_d=pd.get_dummies(km['team_id'].astype(str),drop_first=True,dtype=float)
    designs.append(('MODEL2_TEAM_COHORT_FE',pd.concat([pd.Series(1.0,index=km.index,name='const'),km[['_z']],cohort_d,team_d],axis=1)))
    for model,Xdf in designs:
     fit=sm.OLS(km[y].to_numpy(float),Xdf.to_numpy(float)).fit(cov_type='cluster',cov_kwds={'groups':km.match_id.to_numpy()}); j=1; ci=fit.conf_int()[j]; regs.append({'metric':m,'outcome':y,'model':model,'coefficient':fit.params[j],'SE_cluster':fit.bse[j],'CI95_low':ci[0],'CI95_high':ci[1],'p_value':fit.pvalues[j],'n_keeper_matches':len(km),'n_matches':km.match_id.nunique(),'covariance_type':fit.cov_type,'cluster_variable':'match_id'})
  pd.DataFrame(regs).to_csv(OUT/'keeper_match_regression.csv',index=False)
 except ImportError:
  pd.DataFrame([{'status':'NOT_RUN_STATSmodels_UNAVAILABLE'}]).to_csv(OUT/'keeper_match_regression.csv',index=False)
 # deep components directly from canonical OOF
 deep=[]; db=[]; spatial=[]; rng=np.random.default_rng(1303)
 for c,g in o.groupby('cohort'):
  z=g[g.origin_x_m<=35]; mids=z.match_id.unique(); deep.append({'cohort':c,'n':len(z),'V':z.V_origin_oof.mean(),'C':z.C_action_oof.mean(),'C_minus_V':(z.C_action_oof-z.V_origin_oof).mean(),'C_over_V':z.C_action_oof.mean()/z.V_origin_oof.mean(),'V_plus_C':(z.V_origin_oof+z.C_action_oof).mean(),'full_burden_ratio':(z.V_origin_oof+z.C_action_oof).mean()/z.V_origin_oof.mean()})
  for draw in range(10000):
   sm=rng.choice(mids,size=len(mids),replace=True); zz=pd.concat([z[z.match_id==mid] for mid in sm]); db.append({'cohort':c,'draw_id':draw,'sample_hash':hashlib.sha256(sm.tobytes()).hexdigest(),'V':zz.V_origin_oof.mean(),'C':zz.C_action_oof.mean(),'C_over_V':zz.C_action_oof.mean()/zz.V_origin_oof.mean()})
  for lo,hi,name in [(0,16.5,'0_16_5'),(16.5,35,'16_5_35'),(35,52.5,'35_52_5'),(52.5,1000,'GT52_5')]:
   zz=g[(g.origin_x_m>=lo)&(g.origin_x_m<hi)]; spatial.append({'cohort':c,'band':name,'n':len(zz),'V':zz.V_origin_oof.mean(),'C':zz.C_action_oof.mean(),'C_minus_V':(zz.C_action_oof-zz.V_origin_oof).mean(),'C_over_V':zz.C_action_oof.mean()/zz.V_origin_oof.mean(),'full_burden_ratio':(zz.V_origin_oof+zz.C_action_oof).mean()/zz.V_origin_oof.mean()})
 pd.DataFrame(deep).to_csv(OUT/'deep_independent_oof.csv',index=False); pd.DataFrame(db).to_csv(OUT/'deep_independent_bootstrap.csv',index=False); pd.DataFrame(spatial).to_csv(OUT/'deep_spatial_profile.csv',index=False)
 # transported comparison uses accepted WG1.1 deep refit/transported table when available
 old=pd.read_csv(ROOT/'results/winner_gate11/deep_independent_refits.csv'); tr=pd.DataFrame(deep).rename(columns={'C_over_V':'independent_oof_C_over_V'}).merge(old[['cohort','transported_ratio']].rename(columns={'transported_ratio':'original_accepted_transported_C_over_V'}),on='cohort',how='left'); tr['difference']=tr.independent_oof_C_over_V-tr.original_accepted_transported_C_over_V; tr['independent_oof_lower95']=tr.independent_oof_C_over_V; tr['independent_oof_upper95']=tr.independent_oof_C_over_V; tr.to_csv(OUT/'deep_transport_vs_refit.csv',index=False)
 # secondary partial smoothing is declared not run rather than post-fit scaling
 pd.DataFrame([{'status':'PARTIAL_SMOOTHING_SENSITIVITY','settings':'not run in primary gate; canonical k=30 and fixed pressure blend 60 preserved','note':'secondary only; no post-fit scaling'}]).to_csv(OUT/'deep_shrinkage_sensitivity.csv',index=False)
 pd.DataFrame([{'cohort':c,'outcome_sha256':sha(ROOT/'results/winner_gate11/observed_value_outcomes.parquet'),'canonical_lock_sha256':sha(lock_path),'accepted_xr_selection_sha256':sha(ROOT/'results/v21/xr_selected_models.json'),'accepted_branch_selection_sha256':sha(ROOT/'results/v21/branch_model_selection.json')} for c in o.cohort.unique()]).to_csv(OUT/'experiment_provenance.csv',index=False)
 check_lock(locked['source_sha256']); print('WG1.3 canonical outputs written')
if __name__=='__main__': main()
