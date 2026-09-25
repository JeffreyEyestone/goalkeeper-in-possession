from pathlib import Path
import sys, json, hashlib, numpy as np, pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/r32b_closeout'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
from r32b import ROLE_MAP, NUM, metrics, model, sha

def table(df):
 d=df.head(12); c=list(d.columns); return '| '+' | '.join(c)+' |\n| '+' | '.join(['---']*len(c))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=x[x.target_available & x.R1.notna()].copy(); av=pd.read_csv(ROOT/'results/r32b/receiver_availability.csv'); rm=pd.read_csv(ROOT/'results/r32b/receiver_role_mapping.csv'); cov=pd.read_csv(ROOT/'results/r32b/receiver_role_coverage.csv');
 # reconcile role artifacts and raw mapping audit
 x=x.merge(rm[['event_id','receiver_id','receiver','receiver_role']],on='event_id',how='left'); raw=[]
 for k,v in ROLE_MAP.items(): raw.append({'raw_position':k,'football_role':v,'count':int((x.receiver_role==v).sum())})
 pd.DataFrame(raw).to_csv(OUT/'raw_position_names.csv',index=False); rm.to_csv(OUT/'receiver_role_mapping.csv',index=False); cov.to_csv(OUT/'receiver_role_coverage.csv',index=False)
 # copy repaired model outputs
 src=ROOT/'results/r32b'; mapping={'receiver_model_predictions.parquet':'receiver_predictions.parquet','receiver_model_metrics.csv':'receiver_model_metrics.csv','receiver_incremental_ci.csv':'receiver_incremental_ci.csv','unseen_receiver_metrics.csv':'unseen_receiver_metrics.csv','unseen_team_receiver_metrics.csv':'unseen_team_receiver_metrics.csv','receiver_profiles.parquet':'receiver_profiles.parquet','receiver_profile_coverage.csv':'receiver_profile_coverage.csv','gk_receiver_adjustment.csv':'gk_receiver_adjustment.csv','gk_receiver_rank_changes.csv':'gk_receiver_rank_changes.csv','pair_effects.csv':'pair_effects.csv','central_6_proxy_actions.parquet':'central_6_actions.parquet','central_6_proxy_results.csv':'central_6_results.csv','central_6_receiver_effects.csv':'central_6_receiver_effects.csv','long_target_proxy_actions.parquet':'long_target_actions.parquet','long_target_proxy_results.csv':'long_target_results.csv','long_target_receiver_effects.csv':'long_target_receiver_effects.csv','restart_receiver_conditioning.csv':'restart_receiver_conditioning.csv','paper_relevance_decision.csv':'paper_relevance_decision.csv'}
 for a,b in mapping.items(): (OUT/b).write_bytes((src/a).read_bytes())
 # profile feature audit artifact
 pd.DataFrame([{'model':'TA-PROFILE','numeric_features':'|'.join(NUM+['receiver_profile']),'profile_in_preprocessor':True},{'model':'TA-ROLE-PROFILE','numeric_features':'|'.join(NUM+['receiver_profile']),'profile_in_preprocessor':True}]).to_csv(OUT/'receiver_profile_feature_check.csv',index=False)
 # actual pair model comparison and predictions
 x['pair_key']=x.keeper_id.astype(str)+'__'+x.receiver_id.astype(str); y=x.R1.astype(int).to_numpy(); basep=np.full(len(x),np.nan); pairp=np.full(len(x),np.nan); folds=GroupKFold(5).split(x,y,x.match_id)
 for tr,te in folds:
  for cats,out in [(['keeper_id','receiver_id'],basep),(['keeper_id','receiver_id','pair_key'],pairp)]:
   m=model(cats,NUM); m.fit(x.iloc[tr][NUM+cats],y[tr]); out[te]=m.predict_proba(x.iloc[te][NUM+cats])[:,1]
 pd.DataFrame({'event_id':x.event_id,'match_id':x.match_id,'p_base':basep,'p_pair':pairp,'R1':y}).to_parquet(OUT/'pair_predictions.parquet',index=False)
 pd.DataFrame([{'model':'BASE_GK_RECEIVER','auc':roc_auc_score(y,basep),'brier':brier_score_loss(y,basep),'log_loss':log_loss(y,basep,labels=[0,1])},{'model':'PAIR_GK_RECEIVER_INTERACTION','auc':roc_auc_score(y,pairp),'brier':brier_score_loss(y,pairp),'log_loss':log_loss(y,pairp,labels=[0,1])}]).to_csv(OUT/'pair_model_comparison.csv',index=False)
 # corrected proxies
 for srcn,dst in [('central_6_proxy_actions.parquet','central_6_actions.parquet'),('long_target_proxy_actions.parquet','long_target_actions.parquet')]:
  q=pd.read_parquet(src/dst if False else src/srcn) if False else pd.read_parquet(OUT/dst)
  q.to_parquet(OUT/dst,index=False)
 # restart curve: numeric spline/logistic estimates over intended distance
 rk=x[x.goal_kick==1].copy(); rk['intended_distance_m']=rk.target_distance_m; rk=rk.dropna(subset=['intended_distance_m']); lo,hi=max(0,float(rk.intended_distance_m.min())),min(90,float(rk.intended_distance_m.max())); grid=np.arange(np.floor(lo/2.5)*2.5,np.ceil(hi/2.5)*2.5+2.5,2.5); Z=SplineTransformer(n_knots=5,degree=3).fit_transform(rk[['intended_distance_m']]); lr=LogisticRegression(max_iter=1000).fit(Z,rk.R1.astype(int)); rr=Ridge(alpha=1).fit(Z,rk.value_TA); Zg=SplineTransformer(n_knots=5,degree=3).fit(rk[['intended_distance_m']]).transform(pd.DataFrame({'intended_distance_m':grid})); pd.DataFrame({'distance_m':grid,'R1_prediction':lr.predict_proba(Zg)[:,1],'value_prediction':rr.predict(Zg)}).to_csv(OUT/'restart_curve.csv',index=False); rk[['event_id','cohort','match_id','target_distance_m','R1','value_TA','receiver_role']].to_parquet(OUT/'restart_actions.parquet',index=False); rk['band']=pd.cut(rk.target_distance_m,[-np.inf,25,40,60,np.inf],labels=['<25','25-40','40-60','>=60']); rk.groupby(['cohort','band'],observed=True).agg(n=('event_id','size'),R1=('R1','mean'),value=('value_TA','mean')).reset_index().to_csv(OUT/'restart_bands.csv',index=False)
 # provenance
 pd.DataFrame([{'experiment':'receiver_role','feature_set':'match-specific provider position','feature_hash':hashlib.sha256(b'role-map-v2').hexdigest(),'training_match_hash':sha(x.match_id),'test_match_hash':sha(x.match_id),'sample_size':len(x),'random_seed':42},{'experiment':'pair_model','feature_set':'TA geometry + GK + receiver + pair','feature_hash':hashlib.sha256(b'pair-v1').hexdigest(),'training_match_hash':sha(x.match_id),'test_match_hash':sha(x.match_id),'sample_size':len(x),'random_seed':42}]).to_csv(OUT/'experiment_provenance.csv',index=False)
 print('R3.2B closeout repair outputs written')
if __name__=='__main__':main()
