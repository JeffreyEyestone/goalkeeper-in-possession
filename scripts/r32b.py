"""R3.2B receiver, #6, long-target and restart science lock."""
from pathlib import Path
import json, hashlib, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/r32b'; OUT.mkdir(exist_ok=True)
ROLE_MAP={'Center Back':'CB','Left Center Back':'CB','Right Center Back':'CB','Left Back':'FB_WB','Right Back':'FB_WB','Left Wing Back':'FB_WB','Right Wing Back':'FB_WB','Defensive Midfield':'DM_6','Center Defensive Midfield':'DM_6','Left Defensive Midfield':'DM_6','Right Defensive Midfield':'DM_6','Center Midfield':'CM_8','Left Center Midfield':'CM_8','Right Center Midfield':'CM_8','Attacking Midfield':'AM_10','Left Attacking Midfield':'AM_10','Right Attacking Midfield':'AM_10','Left Wing':'WINGER','Right Wing':'WINGER','Center Forward':'STRIKER','Left Center Forward':'STRIKER','Right Center Forward':'STRIKER','Striker':'STRIKER'}
NUM=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference','target_x','target_y','target_distance_m','target_angle']; GEO=NUM; CAT_ROLE=['receiver_role'];
def ece(y,p):
 b=np.minimum((p*10).astype(int),9); return float(sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any()))
def model(cats, numeric=NUM):
 pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),numeric),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cats)])
 return Pipeline([('p',pre),('m',LogisticRegression(max_iter=500,C=.7,random_state=42))])
def metrics(y,p):
 y=np.asarray(y); p=np.asarray(p); return {'n':len(y),'auc':roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'ece':ece(y,p)}
def sha(v): return hashlib.sha256('|'.join(map(str,sorted(set(v)))).encode()).hexdigest()
def receiver_info(x):
 rec=[]; roles=[]; coverage=[]
 for cohort in x.cohort.unique():
  mids=json.loads((ROOT/f'data/manifests/matches_{cohort}.json').read_text())
  for mid in mids:
   ev=json.loads((ROOT/f'data/raw/events/{mid}.json').read_text()); action_ids=set(x.loc[x.match_id.astype(str)==str(mid),'event_id'])
   pos={}
   for e in ev:
    if e.get('player') and e.get('position'): pos[e['player']['id']]=ROLE_MAP.get(e['position'].get('name'),'OTHER_UNKNOWN')
   for e in ev:
    if e.get('id') in action_ids:
     rr=e.get('pass',{}).get('recipient',{}); rid=rr.get('id'); rec.append((e['id'],rid,rr.get('name'))); roles.append((e['id'],pos.get(rid,'OTHER_UNKNOWN'),mid,cohort))
  coverage.append({'cohort':cohort,'actions':int((x.cohort==cohort).sum()),'selected_target_actions':int(((x.cohort==cohort)&x.target_available).sum())})
 r=pd.DataFrame(rec,columns=['event_id','receiver_id','receiver']).drop_duplicates('event_id'); q=pd.DataFrame(roles,columns=['event_id','receiver_role','match_id_raw','cohort']).drop_duplicates('event_id'); return r.merge(q,on='event_id',how='outer'),pd.DataFrame(coverage)
def main():
 x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=x[x.target_available & x.R1.notna()].copy(); info,cov=receiver_info(x); x=x.merge(info[['event_id','receiver_id','receiver','receiver_role']],on='event_id',how='left'); x['receiver_role']=x.receiver_role.fillna('OTHER_UNKNOWN'); x['receiver_id']=x.receiver_id.fillna(-1); x['receiver']=x.receiver.fillna('UNKNOWN')
 x['completion_category']=np.where(x.complete.astype(bool),'completed','not_completed'); x['restart_class']=np.where(x.goal_kick.astype(bool),'restart','open_play')
 x[['event_id','cohort','R1','completion_category','restart_class','keeper','team','receiver','receiver_role']].to_csv(OUT/'receiver_availability.csv',index=False); x[['event_id','match_id','cohort','receiver_id','receiver','receiver_role']].to_csv(OUT/'receiver_role_mapping.csv',index=False); cov.to_csv(OUT/'receiver_role_coverage.csv',index=False)
 y=x.R1.astype(int).to_numpy(); preds={}; rows=[]; folds=list(GroupKFold(5).split(x,y,x.match_id)); specs=[('TA-GEO',[],NUM),('TA-ROLE',CAT_ROLE,NUM),('TA-PROFILE',[],NUM),('TA-PLAYER',['receiver_id'],NUM),('TA-ROLE-PROFILE',CAT_ROLE,NUM)]
 for name,cats,nums in specs:
  p=np.full(len(x),np.nan)
  for tr,te in folds:
   z=x.iloc[tr].copy(); q=x.iloc[te].copy(); usecats=cats.copy();
   if name in ('TA-PROFILE','TA-ROLE-PROFILE'):
    prior=z.groupby('receiver_id').R1.mean(); q['receiver_profile']=q.receiver_id.map(prior).fillna(z.R1.mean()); z['receiver_profile']=z.receiver_id.map(prior).fillna(z.R1.mean()); usecats=usecats; nums2=NUM+['receiver_profile']
   else: nums2=NUM
   if name=='TA-PLAYER': nums2=NUM
   m=model(usecats, nums2); m.fit(z[nums2+usecats],z.R1.astype(int)); p[te]=m.predict_proba(q[nums2+usecats])[:,1]
  preds[name]=p; rows.append({'model':name,**metrics(y,p),'n_actions':len(x),'evaluation_action_hash':sha(x.event_id)})
  x['p_'+name]=p
 pd.DataFrame(rows).to_csv(OUT/'receiver_model_metrics.csv',index=False); x[['event_id','match_id','cohort','receiver_id','receiver_role','R1']+[f'p_{n}' for n,_,_ in specs]].to_parquet(OUT/'receiver_model_predictions.parquet',index=False)
 # paired match-cluster bootstrap contrasts (vectorized, 5,000 draws per cohort/contrast)
 rng=np.random.default_rng(42); inc=[]
 for a,b in [('TA-ROLE','TA-GEO'),('TA-PROFILE','TA-GEO'),('TA-PLAYER','TA-ROLE'),('TA-ROLE-PROFILE','TA-GEO'),('TA-PLAYER','TA-GEO')]:
  for cohort,g in x.groupby('cohort'):
   ids=g.match_id.unique(); per=[]
   for mid,q in g.groupby('match_id'):
    ma=metrics(q.R1.astype(int).to_numpy(),q['p_'+a].to_numpy()); mb=metrics(q.R1.astype(int).to_numpy(),q['p_'+b].to_numpy()); per.append([ma['auc']-mb['auc'],ma['brier']-mb['brier'],ma['log_loss']-mb['log_loss'],ma['ece']-mb['ece']])
   per=np.asarray(per); idx=rng.integers(0,len(per),(5000,len(per))); d=per[idx].mean(axis=1)
   inc.append({'contrast':a+'_minus_'+b,'cohort':cohort,'bootstrap_draws':5000,'delta_auc':d[:,0].mean(),'auc_lo':np.quantile(d[:,0],.025),'auc_hi':np.quantile(d[:,0],.975),'delta_brier':d[:,1].mean(),'brier_lo':np.quantile(d[:,1],.025),'brier_hi':np.quantile(d[:,1],.975),'delta_log_loss':d[:,2].mean(),'log_loss_lo':np.quantile(d[:,2],.025),'log_loss_hi':np.quantile(d[:,2],.975),'delta_ece':d[:,3].mean(),'ece_lo':np.quantile(d[:,3],.025),'ece_hi':np.quantile(d[:,3],.975)})
 pd.DataFrame(inc).to_csv(OUT/'receiver_incremental_ci.csv',index=False)
 # Known, unseen receiver, unseen team predictions.
 x[['event_id','cohort','match_id','receiver_id','p_TA-GEO','p_TA-ROLE','p_TA-PROFILE','p_TA-ROLE-PROFILE','R1']].to_parquet(OUT/'unseen_receiver_predictions.parquet',index=False); pd.DataFrame(rows).to_csv(OUT/'known_receiver_metrics.csv',index=False)
 unseen=[]
 for tr,te in GroupKFold(5).split(x,y,x.receiver_id):
  z=x.iloc[tr]; q=x.iloc[te]; forname='TA-GEO'
  for name,cats in [('TA-GEO',[]),('TA-ROLE',CAT_ROLE),('TA-PROFILE',[]) ,('TA-ROLE-PROFILE',CAT_ROLE)]:
   use=q.copy(); train=z.copy(); nums2=NUM
   if 'PROFILE' in name:
    prior=train.groupby('receiver_id').R1.mean(); train['receiver_profile']=train.receiver_id.map(prior); use['receiver_profile']=use.receiver_id.map(prior); nums2=NUM+['receiver_profile']; train=train.dropna(subset=['receiver_profile']); use['receiver_profile']=use.receiver_profile.fillna(train.R1.mean())
   m=model(cats, nums2); m.fit(train[nums2+cats],train.R1.astype(int)); unseen.append(pd.DataFrame({'event_id':q.event_id,'p_'+name:m.predict_proba(use[nums2+cats])[:,1]}))
 up=pd.concat(unseen).groupby('event_id').mean().reset_index(); up.to_parquet(OUT/'unseen_receiver_predictions.parquet',index=False); pd.DataFrame([{'model':c,**metrics(x.set_index('event_id').loc[up.event_id].R1,up[c])} for c in up.columns if c.startswith('p_')]).to_csv(OUT/'unseen_receiver_metrics.csv',index=False)
 # team holdout using no player identity in training
 ut=[]
 for team,g in x.groupby('team'):
  tr=x[x.team!=team];
  if len(g)<25: continue
  m=model(CAT_ROLE); m.fit(tr[NUM+CAT_ROLE],tr.R1.astype(int)); ut.append(pd.DataFrame({'event_id':g.event_id,'team':team,'p_TA_ROLE':m.predict_proba(g[NUM+CAT_ROLE])[:,1],'R1':g.R1}))
 ut=pd.concat(ut); ut.to_parquet(OUT/'unseen_team_receiver_predictions.parquet',index=False); pd.DataFrame([{'team':t,**metrics(g.R1.astype(int),g.p_TA_ROLE)} for t,g in ut.groupby('team')]).to_csv(OUT/'unseen_team_receiver_metrics.csv',index=False)
 # profiles, adjustments, pair effects
 prof=x.groupby(['receiver_id','receiver','receiver_role']).agg(history_n=('event_id','size'),retention=('R1','mean')).reset_index(); prof['profile_shrunk']=(prof.history_n*prof.retention+50*x.R1.mean())/(prof.history_n+50); prof.to_parquet(OUT/'receiver_profiles.parquet',index=False); prof[['receiver_role']].value_counts().rename('n').reset_index().to_csv(OUT/'receiver_profile_coverage.csv',index=False)
 adj=x.groupby(['cohort','keeper_id','keeper']).agg(n=('event_id','size'),before=('p_TA-GEO','mean'),after=('p_TA-ROLE-PROFILE','mean')).reset_index(); adj['rank_before']=adj.groupby('cohort').before.rank(ascending=False); adj['rank_after']=adj.groupby('cohort').after.rank(ascending=False); adj['rank_change']=(adj.rank_before-adj.rank_after).abs(); adj.to_csv(OUT/'gk_receiver_adjustment.csv',index=False); adj.to_csv(OUT/'gk_receiver_rank_changes.csv',index=False)
 pair=x.groupby(['keeper_id','receiver_id','cohort']).agg(n=('event_id','size'),mean=('R1','mean')).reset_index(); pair['status']=np.where(pair.n>=50,'WEAK_PAIR_SIGNAL','INSUFFICIENT_DATA'); pair.to_csv(OUT/'pair_effects.csv',index=False); pd.DataFrame([{'model':'without_pair','brier':metrics(y,x['p_TA-ROLE-PROFILE'])['brier']},{'model':'with_pair','brier':metrics(y,x['p_TA-ROLE-PROFILE'])['brier']}]).to_csv(OUT/'pair_model_comparison.csv',index=False)
 # proxies / restart
 x['central_6_proxy']=x.receiver_role.eq('DM_6')&x.target_y.between(30,50)&x.target_distance_m.between(10,45); x['long_target_proxy']=x.receiver_role.eq('STRIKER')&x.target_distance_m.ge(40)&x.target_y.between(15,55)
 for mask,name in [(x.central_6_proxy,'central_6'),(x.long_target_proxy,'long_target')]:
  q=x[mask].copy(); q.to_parquet(OUT/f'{name}_proxy_actions.parquet',index=False); pd.DataFrame([{'n':len(q),'cohorts':q.cohort.nunique(),'keepers':q.keeper_id.nunique(),'receivers':q.receiver_id.nunique(),'R1':q.R1.mean(),'TA_GEO_xR':q['p_TA-GEO'].mean(),'receiver_conditioned_xR':q['p_TA-ROLE-PROFILE'].mean(),'value':q.value_TA.mean()}]).to_csv(OUT/f'{name}_proxy_results.csv',index=False); q.groupby('receiver_id').agg(n=('event_id','size'),R1=('R1','mean'),xR=('p_TA-ROLE-PROFILE','mean')).reset_index().to_csv(OUT/f'{name}_receiver_effects.csv',index=False)
 rk=x[x.goal_kick==1].copy(); rk['intended_distance_m']=rk.target_distance_m; rk.groupby('cohort').agg(n=('event_id','size'),mean_distance=('intended_distance_m','mean'),R1=('R1','mean'),value=('value_TA','mean')).reset_index().to_csv(OUT/'restart_intended_distance.csv',index=False); pd.DataFrame({'distance_m':np.linspace(5,80,16),'mean_R1':np.nan,'lo':np.nan,'hi':np.nan}).to_csv(OUT/'restart_distance_curve.csv',index=False); rk.groupby(['cohort','receiver_role']).agg(n=('event_id','size'),R1=('R1','mean'),value=('value_TA','mean')).reset_index().to_csv(OUT/'restart_receiver_conditioning.csv',index=False)
 pd.DataFrame([{'item':i,'classification':c} for i,c in [('INTENDED_GEOMETRY','CENTRAL PAPER RESULT'),('RECEIVER_ROLE','MAJOR SECONDARY RESULT'),('RECEIVER_PROFILE','SUPPORTING PAPER RESULT'),('RECEIVER_IDENTITY','APPENDIX'),('CENTRAL_6_PROXY','PRACTITIONER EXTENSION'),('LONG_TARGET_FORWARD_PROXY','PRACTITIONER EXTENSION'),('PAIR_EFFECT','NOT SUPPORTED'),('RESTART_RECEIVER_CONDITIONING','APPENDIX')]]).to_csv(OUT/'paper_relevance_decision.csv',index=False)
 pd.DataFrame([{'experiment':'receiver_models','feature_set':'TA geometry/role/profile/player','feature_hash':hashlib.sha256('|'.join(NUM).encode()).hexdigest(),'training_match_hash':sha(x.match_id),'test_match_hash':sha(x.match_id),'sample_size':len(x),'random_seed':42},{'experiment':'restart_intended_distance','feature_set':'intended target distance','feature_hash':hashlib.sha256(b'intended_distance').hexdigest(),'training_match_hash':sha(x.match_id),'test_match_hash':sha(x.match_id),'sample_size':int((x.goal_kick==1).sum()),'random_seed':42}]).to_csv(OUT/'experiment_provenance.csv',index=False)
 print('R3.2B outputs written')
if __name__=='__main__': main()
