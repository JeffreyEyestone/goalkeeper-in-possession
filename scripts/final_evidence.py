from pathlib import Path
import json, hashlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/final_evidence'; OUT.mkdir(exist_ok=True)
COMMIT='FINAL_EVIDENCE_LOCK_SOURCE'
def tab(d):
 d=d.head(15); c=list(d.columns); return '| '+' | '.join(c)+' |\n| '+' | '.join(['---']*len(c))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def bootstrap_metric(a,b,y,matches,n=10000,seed=42):
 rng=np.random.default_rng(seed); rows=[]
 for m in pd.unique(matches):
  z=matches==m; yy=np.asarray(y)[z]; aa=np.asarray(a)[z]; bb=np.asarray(b)[z]
  rows.append([roc_auc_score(yy,aa)-roc_auc_score(yy,bb) if len(np.unique(yy))>1 else 0.0,brier_score_loss(yy,aa)-brier_score_loss(yy,bb),log_loss(yy,aa,labels=[0,1])-log_loss(yy,bb,labels=[0,1])])
 rows=np.asarray(rows); idx=rng.integers(0,len(rows),(n,len(rows))); return rows[idx].mean(axis=1)
def main():
 v=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=v[v.R1.notna()].copy(); cohorts=sorted(x.cohort.unique())
 # Deep consequence, predefined deep origin corridor x<35.
 deep=x[x.origin_x<35].copy(); deep['ratio']=deep.failure_cost_TF/deep.origin_value_TF.replace(0,np.nan)
 rows=[]; boots=[]; rng=np.random.default_rng(42)
 for c,g in deep.groupby('cohort'):
  rows.append({'cohort':c,'n_actions':len(g),'n_matches':g.match_id.nunique(),'mean_origin_value':g.origin_value_TF.mean(),'mean_failure_consequence':g.failure_cost_TF.mean(),'ratio':g.failure_cost_TF.mean()/g.origin_value_TF.mean()})
  mids=np.array(g.match_id.unique()); sums=g.groupby('match_id').agg(f=('failure_cost_TF','sum'),o=('origin_value_TF','sum')); idx=rng.integers(0,len(sums),(10000,len(sums))); fa=sums.f.to_numpy(); oa=sums.o.to_numpy(); vals=fa[idx].sum(axis=1)/oa[idx].sum(axis=1)
  boots.extend({'cohort':c,'draw':i,'ratio':z} for i,z in enumerate(vals))
 pd.DataFrame(rows).to_csv(OUT/'deep_consequence_final.csv',index=False); pd.DataFrame(boots).to_csv(OUT/'deep_consequence_bootstrap.csv',index=False)
 # Completion/value by cohort and old comparison.
 cr=[]
 for c,g in x.groupby('cohort'):
  q=g.groupby(['keeper_id','keeper']).agg(n=('event_id','size'),matches=('match_id','nunique'),completion=('complete','mean'),value=('value_TF','mean')).query('n>=25')
  rho=spearmanr(q.completion,q.value).statistic if len(q)>2 else np.nan; cr.append({'cohort':c,'eligible_keepers':len(q),'spearman_completion_value':rho,'bootstrap_lo':np.nan,'bootstrap_hi':np.nan})
 pd.DataFrame(cr).to_csv(OUT/'completion_value_final.csv',index=False)
 # Accepted LOCO portability.
 lf=pd.read_csv(ROOT/'results/r32a_final/loco_metrics.csv'); lf=lf[lf.held_out_cohort.notna() & lf.scenario.isna()] if 'scenario' in lf else lf
 lf[['held_out_cohort','n','auc','brier','calibration_slope','gap']].rename(columns={'held_out_cohort':'cohort','calibration_slope':'slope','gap':'mean_prediction_gap'}).to_csv(OUT/'context_final.csv',index=False)
 # local calibration accepted patch summary
 pd.read_csv(ROOT/'results/r32a_patch/loco_local_calibration_summary.csv').to_csv(OUT/'local_calibration_final.csv',index=False)
 # TF vs TA-GEO selected common sample from R3.2B closeout predictions.
 rp=pd.read_parquet(ROOT/'results/r32b_closeout/receiver_predictions.parquet'); q=x[x.event_id.isin(rp.event_id)].merge(rp[['event_id','p_TA-GEO','p_TA-ROLE','p_TA-PLAYER']],on='event_id'); q['p_TF']=q.rho_TF
 overall=[]
 for name in ['p_TF','p_TA-GEO']:
  p=q[name].to_numpy(); overall.append({'model':name,'n':len(q),'auc':roc_auc_score(q.R1,p),'brier':brier_score_loss(q.R1,p),'log_loss':log_loss(q.R1,p,labels=[0,1])})
 pd.DataFrame(overall).to_csv(OUT/'tf_ta_geo_final.csv',index=False); d=bootstrap_metric(q.p_TF.to_numpy(),q['p_TA-GEO'].to_numpy(),q.R1.to_numpy(),q.match_id.to_numpy()); pd.DataFrame(d,columns=['delta_auc','delta_brier','delta_log_loss']).to_csv(OUT/'tf_ta_geo_bootstrap.csv',index=False)
 # Receiver supporting point results and proper bootstrap overall.
 sup=[]
 for a,b in [('p_TA-ROLE','p_TA-GEO'),('p_TA-PLAYER','p_TA-GEO')]:
  d=bootstrap_metric(q[a].to_numpy(),q[b].to_numpy(),q.R1.to_numpy(),q.match_id.to_numpy()); sup.append({'contrast':a+'_minus_'+b,'delta_auc':d[:,0].mean(),'auc_lo':np.quantile(d[:,0],.025),'auc_hi':np.quantile(d[:,0],.975),'delta_brier':d[:,1].mean(),'brier_lo':np.quantile(d[:,1],.025),'brier_hi':np.quantile(d[:,1],.975),'delta_log_loss':d[:,2].mean(),'log_loss_lo':np.quantile(d[:,2],.025),'log_loss_hi':np.quantile(d[:,2],.975),'draws':10000})
 pd.DataFrame(sup).to_csv(OUT/'receiver_supporting_final.csv',index=False)
 # restart corrected bands from closeout actions.
 rr=pd.read_parquet(ROOT/'results/r32b_closeout/restart_actions.parquet'); rr['band']=pd.cut(rr.target_distance_m,[-np.inf,25,40,60,np.inf],labels=['<25','25-40','40-60','>=60']); pd.concat([rr.groupby(['cohort','band'],observed=True).agg(n=('event_id','size'),value=('value_TA','mean'),R1=('R1','mean')).reset_index()]).to_csv(OUT/'restart_claim_final.csv',index=False)
 # repeatability descriptive from two league halves.
 rep=[]
 for c in ['PL1516','WSL2021']:
  g=x[x.cohort==c].copy(); g['half_split']=g.match_id.astype(str).map(lambda z:hash(z)%2); a=g[g.half_split==0].groupby('keeper_id').R1.mean(); b=g[g.half_split==1].groupby('keeper_id').R1.mean(); z=pd.concat([a,b],axis=1).dropna(); rep.append({'cohort':c,'eligible_keepers':len(z),'R1_spearman':z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),'R1_pearson':z.iloc[:,0].corr(z.iloc[:,1]),'interpretation':'exploratory; not stable execution skill'})
 pd.DataFrame(rep).to_csv(OUT/'repeatability_final.csv',index=False)
 # claim ledger
 claims=[]
 def add(cid,text,est,status,level,reason,source,code,pop='accepted analyses'):
  claims.append({'claim_id':cid,'plain_english_claim':text,'estimand':text,'population':pop,'sample_size':int(len(x)),'cohorts':','.join(cohorts),'statistic':'see canonical table','estimate':est,'lower_95':np.nan,'upper_95':np.nan,'status':status,'paper_level':level,'reason':reason,'canonical_source':source,'generating_code':code,'git_provenance':COMMIT})
 add('C1','Completion alone is insufficient to value goalkeeper distribution','see completion_value_final','QUALIFIED','MAJOR','completion/value associations vary and are not a complete value model','completion_value_final.csv','scripts/final_evidence.py')
 add('C2','Deep failure consequence is large relative to origin value','see deep_consequence_final','QUALIFIED','CENTRAL','descriptive match bootstrap','deep_consequence_final.csv','scripts/final_evidence.py')
 add('C4','Relative TF risk structure has moderate cross-environment portability','see context_final','QUALIFIED','CENTRAL','LOCO AUC survives with probability-scale drift','context_final.csv','scripts/final_evidence.py')
 add('C5','Absolute TF probability transfers unchanged','rejected','FAIL','LIMITATION','gaps and calibration drift remain','context_final.csv','scripts/final_evidence.py')
 add('C6','Competition context can change unseen-team probability behavior','see accepted bootstrap','QUALIFIED','MAJOR','associative context result','results/r32a_closeout/team_context_bootstrap_summary.csv','scripts/final_evidence.py')
 add('C7','Local calibration improves probability reporting but does not solve uncertainty','see local_calibration_final','QUALIFIED','MAJOR','1/3/5 match intervals remain wide','local_calibration_final.csv','scripts/final_evidence.py')
 add('C8','Probability adaptation can alter xT-GK valuation','see R3.2A closeout','QUALIFIED','MAJOR','accepted canonical xT sensitivity','results/r32a_closeout/xtgk_action_sensitivity.csv','scripts/r32a_closeout.py')
 add('C10','Intended geometry adds beyond TF on selected actions','see tf_ta_geo_final','QUALIFIED','MAJOR','outcome-dependent selected sample','tf_ta_geo_final.csv','scripts/final_evidence.py')
 add('C11','Receiver role adds supporting information beyond geometry','see receiver_supporting_final','QUALIFIED','SUPPORTING','selected-sample OOF evidence','receiver_supporting_final.csv','scripts/final_evidence.py')
 add('C12','Specific receiver identity adds supporting point information','see receiver_supporting_final','QUALIFIED','SUPPORTING','no player-transfer claim','receiver_supporting_final.csv','scripts/final_evidence.py')
 add('C13','Receiver profile is validated as a positive main result','rejected','FAIL','LIMITATION','methodological limitations and no accepted positive claim','results/r32b_closeout/receiver_profile_feature_check.csv','scripts/final_evidence.py')
 add('C14','Pair identity improves aggregate held-out prediction','rejected','FAIL','APPENDIX','pair result is appendix/future work only','results/r32b_closeout/pair_model_comparison.csv','scripts/final_evidence.py')
 add('C15','Central #6 proxy is validated paper evidence','rejected','EXPLORATORY','FUTURE_WORK','proxy requires richer option-set data','results/r32b_closeout/central_6_results.csv','scripts/final_evidence.py')
 add('C16','Long target-forward proxy is validated paper evidence','rejected','EXPLORATORY','FUTURE_WORK','proxy requires richer option-set data','results/r32b_closeout/long_target_results.csv','scripts/final_evidence.py')
 add('C17','Historical universal 40–60 m restart claim survives','rejected','FAIL','LIMITATION','corrected intended-target bands do not validate it','restart_claim_final.csv','scripts/final_evidence.py')
 add('C18','Residual execution skill is stable across leagues','rejected','EXPLORATORY','APPENDIX','split-half evidence is exploratory','repeatability_final.csv','scripts/final_evidence.py')
 add('C19','xT-GK is style-neutral','rejected','FAIL','LIMITATION','not supported as a universal claim','results/r32a_closeout/xtgk_action_sensitivity.csv','scripts/final_evidence.py')
 add('C20','The model predicts transfer success across clubs/systems','rejected','FAIL','LIMITATION','no causal transfer experiment','results/r32a_closeout/team_context_bootstrap_summary.csv','scripts/final_evidence.py')
 add('C21','Consequence structure is more stable than probability scale','supported synthesis','QUALIFIED','CENTRAL','descriptive contrast across accepted analyses','deep_consequence_final.csv','scripts/final_evidence.py')
 add('C22','Distribution should be valued as a two-branch ex-ante expectation','canonical framework','PASS','CENTRAL','canonical scientific definition','src/gkpossession/xtgk.py','scripts/final_evidence.py')
 ledger=pd.DataFrame(claims); ledger.to_csv(OUT/'claim_ledger.csv',index=False)
 facts=[{'fact_id':'F1','claim_id':'C2','text_label':'Deep failure-consequence ratio is elevated across cohorts','value':float(pd.DataFrame(rows).ratio.mean()),'unit':'ratio','sample':int(len(deep)),'cohort/population':'five cohorts, deep origin zone','status':'QUALIFIED','canonical_source':'deep_consequence_final.csv','generating_code':'scripts/final_evidence.py','git_commit':COMMIT},{'fact_id':'F2','claim_id':'C4','text_label':'LOCO AUC range','value':[float(lf.auc.min()),float(lf.auc.max())],'unit':'AUC range','sample':int(lf.n.sum()),'cohort/population':'five held-out cohorts','status':'QUALIFIED','canonical_source':'context_final.csv','generating_code':'scripts/final_evidence.py','git_commit':COMMIT},{'fact_id':'F3','claim_id':'C10','text_label':'TA-GEO selected-sample gain','value':float(overall[1]['auc']-overall[0]['auc']),'unit':'delta AUC','sample':int(len(q)),'cohort/population':'target-identifiable selected sample','status':'QUALIFIED','canonical_source':'tf_ta_geo_final.csv','generating_code':'scripts/final_evidence.py','git_commit':COMMIT}]
 json.dump(facts,open(OUT/'abstract_fact_set.json','w'),indent=2); json.dump([dict(r) for r in claims],open(OUT/'paper_fact_set.json','w'),indent=2)
 fig=pd.DataFrame([{'figure':'Figure 1','topic':'two-branch xT-GK framework','status':'CENTRAL','source':'canonical framework'},{'figure':'Figure 2','topic':'deep consequence ratio with 10,000 match bootstrap','status':'CENTRAL','source':'deep_consequence_bootstrap.csv'},{'figure':'Figure 3','topic':'TF vs TA-GEO selected sample','status':'MAJOR','source':'tf_ta_geo_bootstrap.csv'},{'figure':'Figure 4','topic':'LOCO portability/calibration','status':'MAJOR','source':'context_final.csv'},{'figure':'Figure 5','topic':'local calibration/xT sensitivity','status':'MAJOR','source':'local_calibration_final.csv'},{'figure':'Figure 6','topic':'supporting receiver role point results','status':'SUPPORTING','source':'receiver_supporting_final.csv'}]); fig.to_csv(OUT/'figure_plan.csv',index=False)
 print('final evidence written')
if __name__=='__main__':main()
