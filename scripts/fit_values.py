import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import pandas as pd,numpy as np,joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.features import matrix
from gkpossession.surfaces import fit_bundle,lookup,zone
from gkpossession.failure_cost import ShrunkBranchMean,regressor
from gkpossession.xtgk import score
from gkpossession.xr import fit_calibration,calibrated
from gkpossession.stats import metrics

def model_for(kind,est):return ShrunkBranchMean(est) if kind=='shrunk' else regressor(est)
def main():
    x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet');criteria=pd.read_parquet(ROOT/'data/processed/criteria.parquet');x=x.merge(criteria,on='event_id',validate='one_to_one')
    states=pd.read_parquet(ROOT/'data/processed/surface_states.parquet');split=json.loads((ROOT/'config/wc2018_split.json').read_text());surf=fit_bundle(states,split['train']);out=ROOT/'results/v21';models=ROOT/'data/processed/models'
    joblib.dump(surf,models/'surface.joblib');branch_metrics=[];choices={};local_metrics=[];local_splits={}
    valid=x.R1.notna() & x.R1_state_x.notna() & x.R1_state_y.notna()
    x['observed_success_payoff']=np.nan;x['observed_failure_cost']=np.nan
    own=valid & (x.R1==1);lost=valid & (x.R1==0)
    x.loc[own,'observed_success_payoff']=lookup(surf,x.loc[own,'R1_state_x'],x.loc[own,'R1_state_y'],pressure=x.loc[own,'R1_state_pressure'])
    x.loc[lost,'observed_failure_cost']=lookup(surf,x.loc[lost,'R1_state_x'],x.loc[lost,'R1_state_y'],name='T')
    for est in ['TF','TA']:
        allg=x if est=='TF' else x[x.target_available]
        train=allg[allg.partition=='train'];branch_models={}
        for label,target in [(1,'observed_success_payoff'),(0,'observed_failure_cost')]:
            trainbranch=train[(train.R1==label)&train[target].notna()];X=matrix(trainbranch,est);y=trainbranch[target];cv={}
            for kind in ['shrunk','boosted']:
                errors=[]
                for a,b in GroupKFold(4).split(X,y,trainbranch.match_id):
                    model=model_for(kind,est);model.fit(X.iloc[a],y.iloc[a]);pred=np.maximum(model.predict(X.iloc[b]),0);errors.extend((pred-y.iloc[b].to_numpy()).tolist())
                cv[kind]=float(np.sqrt(np.mean(np.square(errors))))
            chosen='boosted' if cv['boosted']<.95*cv['shrunk'] else 'shrunk'
            model=model_for(chosen,est).fit(X,y);branch_models[label]=model;choices[f'{est}_{label}']={'family':chosen,'training_n':len(y),'training_matches':trainbranch.match_id.nunique(),'groupcv_rmse':cv,'rule':'prefer shrinkage unless boosted RMSE improves by at least 5%'}
            for c,g in allg[(allg.R1==label)&allg[target].notna()&~allg.partition.isin(['train','calibration'])].groupby('cohort'):
                pred=np.maximum(model.predict(matrix(g,est)),0);branch_metrics.append({'estimand':est,'branch':'success' if label else 'failure','cohort':c,'n':len(g),'mae':mean_absolute_error(g[target],pred),'rmse':np.sqrt(mean_squared_error(g[target],pred)),'target_mean':g[target].mean(),'predicted_mean':pred.mean(),'target':'frozen surface proxy at observed R1 resolution'})
        joblib.dump(branch_models,models/f'branches_{est}.joblib')
        val=score(allg,allg[f'rho_{est}'],branch_models[1],branch_models[0],surf,est)
        for k,v in val.items():x.loc[allg.index,k+'_'+est]=v
        x.loc[allg.index,f'value_{est}_platt']=score(allg,allg[f'{est}_logistic_platt'],branch_models[1],branch_models[0],surf,est)['value']
        # Secondary, two-fold match-cross-fitted Platt recalibration within each external population.
        raw_model=joblib.load(models/f'xr_{est}.joblib')['model'];x[f'rho_{est}_local']=np.nan
        for c,g in allg.groupby('cohort'):
            if c=='WC2018':x.loc[g.index,f'rho_{est}_local']=g[f'rho_{est}'];continue
            ids=np.array(sorted(g.match_id.unique()));np.random.default_rng(42).shuffle(ids);halfsets=[ids[::2],ids[1::2]];local_splits[f'{est}_{c}']=[a.tolist() for a in halfsets]
            for half in [0,1]:
                calg=g[g.match_id.isin(halfsets[1-half])&g.R1.notna()];testg=g[g.match_id.isin(halfsets[half])]
                assert not set(calg.match_id)&set(testg.match_id)
                mapping=fit_calibration(calg.R1.astype(int),raw_model.predict_proba(matrix(calg,est))[:,1],'platt')
                x.loc[testg.index,f'rho_{est}_local']=calibrated(mapping,raw_model.predict_proba(matrix(testg,est))[:,1],'platt')
            evalg=x.loc[g.index];evalg=evalg[evalg.R1.notna()];local_metrics.append({'estimand':est,'cohort':c,'scope':'secondary local two-fold match cross-calibration',**metrics(evalg.R1,evalg[f'rho_{est}_local'])})
        x.loc[allg.index,f'value_{est}_local']=score(allg,x.loc[allg.index,f'rho_{est}_local'],branch_models[1],branch_models[0],surf,est)['value']
        x.loc[allg.index,f'roe_{est}']=allg.R1-allg[f'rho_{est}'];x.loc[allg.index,f'roe_{est}_local']=allg.R1-x.loc[allg.index,f'rho_{est}_local']
        for kappa in [1.25,1.5,1.75,2]:x.loc[allg.index,f'value_{est}_k{kappa}']=score(allg,allg[f'rho_{est}'],branch_models[1],branch_models[0],surf,est,kappa=kappa)['value']
        print(est,'branches valued',flush=True)
    v0=lookup(surf,x.origin_x,x.origin_y);vend=lookup(surf,x.end_x,x.end_y);oppend=lookup(surf,120-x.end_x,80-x.end_y,name='T')
    x['baseline_delta']=vend-v0;x['baseline_fixed_cost']=vend-v0-float(surf['T'].mean());x['baseline_location_cost']=vend-v0-oppend
    x['criterion_resolution']=np.where(x.R1==1,x.observed_success_payoff,-x.observed_failure_cost)-lookup(surf,x.origin_x,x.origin_y,pressure=x.origin_pressure)
    # Standard grid xT baseline: shoot probability + completed move transition value, WC18 train only.
    st=states[states.match_id.isin(split['train']) & states.etype.isin(['Pass','Carry','Shot'])].copy();z=zone(st.x,st.y);dest=zone(st.end_x,st.end_y);n=192
    total=np.bincount(z,minlength=n);shot_reward=np.bincount(z,weights=st.shot_xg,minlength=n)/np.maximum(total,1)
    transition=np.zeros((n,n));moving=st.etype.ne('Shot').to_numpy();success=(st.etype.eq('Carry')|st.complete.eq(1)).to_numpy()
    np.add.at(transition,(z[moving&success],dest[moving&success]),1);transition/=np.maximum(total[:,None],1)
    xt=np.zeros(n)
    for _ in range(2000):
        new=shot_reward+transition@xt
        if np.max(abs(new-xt))<1e-10:break
        xt=new
    np.save(models/'standard_xt.npy',xt);x['baseline_standard_xt']=x.complete*(xt[zone(x.end_x,x.end_y)]-xt[zone(x.origin_x,x.origin_y)])
    # Historical ablation aligns by match and event index; no missing values fabricated.
    hist=[]
    for p in (ROOT/'historical/goalkeeper-in-possession-repo/out').glob('*_valued.parquet'):
        h=pd.read_parquet(p);hist.append(h[['match_id','idx','xtgk']].rename(columns={'idx':'event_index','xtgk':'baseline_v20'}))
    x=x.merge(pd.concat(hist),on=['match_id','event_index'],how='left',validate='one_to_one')
    x.to_parquet(ROOT/'data/processed/valued_actions.parquet',index=False)
    pd.DataFrame(branch_metrics).to_csv(out/'branch_model_metrics.csv',index=False);pd.DataFrame(local_metrics).to_csv(out/'local_calibration_metrics.csv',index=False)
    (out/'branch_model_selection.json').write_text(json.dumps(choices,indent=2));(ROOT/'config/local_calibration_splits.json').write_text(json.dumps(local_splits,indent=2))
if __name__=='__main__':main()
