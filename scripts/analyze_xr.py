import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.stats import metrics,bootstrap_metrics,calibration
x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet');out=ROOT/'results/v21';selected=json.loads((out/'xr_selected_models.json').read_text());xr=pd.read_csv(out/'xr_metrics.csv')
reps=json.loads((ROOT/'config/analysis.json').read_text())['bootstrap_replicates']
for i,r in xr.iterrows():
    if r.family!=selected[r.estimand]['family'] or pd.notna(r.get('auc_lo',np.nan)):continue
    g=x[(x.cohort==r.cohort)&x.R1.notna()&~x.partition.isin(['train','calibration']) & (True if r.estimand=='TF' else x.target_available)].copy()
    pcol=f'{r.estimand}_{r.family}_{r.calibration}';ci=bootstrap_metrics(g,pcol,reps=reps)
    for k,(lo,hi) in ci.items():xr.loc[i,k+'_lo']=lo;xr.loc[i,k+'_hi']=hi
    print(r.estimand,r.calibration,r.cohort,'bootstrap done',flush=True)
xr.to_csv(out/'xr_metrics.csv',index=False)
boundary=[];clipping=[];overlap=[];keepers=[];rank=[];repeat=[]
eval=x[x.R1.notna() & ~x.partition.isin(['train','calibration'])].copy()
for c,g in eval.groupby('cohort'):
    p=g.rho_TF.to_numpy();y=g.R1.to_numpy()
    boundary.append({'cohort':c,'n':len(g),'p_zero_n':int(sum(p==0)),'positive_at_zero':int(sum((p==0)&(y==1))),'p_one_n':int(sum(p==1)),'negative_at_one':int(sum((p==1)&(y==0))),'raw_log_loss':'infinite' if np.any(((p==0)&(y==1))|((p==1)&(y==0))) else 'finite'})
    for eps in [.01,.001,.0001,.00001,.000001]:
        # Use explicit eps here; the main calibration function pins a 1e-5 floor.
        from scipy.special import logit,expit
        from scipy.optimize import minimize
        z=logit(np.clip(p,eps,1-eps));X=np.column_stack([np.ones(len(y)),z]);fit=minimize(lambda b:np.sum(np.logaddexp(0,X@b)-y*(X@b)),[0,1],jac=lambda b:X.T@(expit(X@b)-y),method='BFGS')
        clipping.append({'cohort':c,'epsilon':eps,'intercept':fit.x[0],'slope':fit.x[1]})
    common=g[g.target_available]
    for estimand in ['TF','TA']:
        for method in ['isotonic','platt']:
            pcol=f'{estimand}_{selected[estimand]["family"]}_{method}'
            overlap.append({'cohort':c,'estimand':estimand,'calibration':method,**metrics(common.R1,common[pcol])})
    for (kid,kname),kg in common.groupby(['keeper_id','keeper']):
        if len(kg)<(200 if c in ['PL1516','WSL2021'] else 25) or kg.match_id.nunique()<2:continue
        keepers.append({'cohort':c,'keeper_id':kid,'keeper':kname,'n':len(kg),'matches':kg.match_id.nunique(),'retention':kg.R1.mean(),'rho_TF':kg.rho_TF.mean(),'rho_TA':kg.rho_TA.mean(),'roe_TF':(kg.R1-kg.rho_TF).mean(),'roe_TA':(kg.R1-kg.rho_TA).mean()})
k=pd.DataFrame(keepers);k.to_csv(out/'tf_ta_overlap_keepers.csv',index=False)
for c,g in k.groupby('cohort'):
    for metric in ['rho','roe']:
        rank.append({'cohort':c,'metric':metric,'keepers':len(g),'spearman':spearmanr(g[metric+'_TF'],g[metric+'_TA']).statistic,'pearson':pearsonr(g[metric+'_TF'],g[metric+'_TA']).statistic})
for c in ['PL1516','WSL2021']:
    g=eval[eval.cohort==c].copy();ids=sorted(g.match_id.unique());rng=np.random.default_rng(42);rng.shuffle(ids);half={mid:i%2 for i,mid in enumerate(ids)};g['half']=g.match_id.map(half)
    for sample,sg in [('all',g),('common',g[g.target_available])]:
        for estimand in (['TF'] if sample=='all' else ['TF','TA']):
            sg=sg.copy();sg['rho']=sg[f'rho_{estimand}'];sg['roe']=sg.R1-sg.rho
            for metric in ['complete','R1','rho','roe']:
                tab=sg.groupby(['keeper_id','half'])[metric].agg(['mean','count']).unstack('half').dropna();tab=tab[(tab['count'][0]>=100)&(tab['count'][1]>=100)]
                if len(tab)>=3:
                    a,b=tab['mean'][0],tab['mean'][1];r=pearsonr(a,b).statistic;s=spearmanr(a,b).statistic
                    repeat.append({'cohort':c,'sample':sample,'estimand':estimand,'metric':metric,'keepers':len(tab),'pearson':r,'spearman':s,'spearman_brown':2*r/(1+r)})
for name,rows in [('probability_boundaries',boundary),('calibration_slope_clip_sensitivity',clipping),('tf_ta_overlap_metrics',overlap),('tf_ta_probability_rank_comparison',rank),('preliminary_xr_repeatability',repeat)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
# Preserve predeclared gates, updating intervals to the configured 500 match bootstraps.
gates=[]
for r in xr.itertuples():
    if r.family!=selected[r.estimand]['family'] or r.calibration!='isotonic' or r.cohort=='WC2018':continue
    if r.ece>.10 or abs(r.mean_gap)>.10 or r.calibration_slope_hi<.5 or r.calibration_slope_lo>2:
        gates.append({'gate':'external_calibration','estimand':r.estimand,'cohort':r.cohort,'slope':r.calibration_slope,'slope_lo':r.calibration_slope_lo,'slope_hi':r.calibration_slope_hi,'ece':r.ece,'mean_gap':r.mean_gap,'note':'Slope threshold depends on declared logit clipping; inspect boundary predictions and sensitivity before interpretation.'})
(out/'scientific_gates.json').write_text(json.dumps(gates,indent=2))
print('GATES',gates,flush=True)
