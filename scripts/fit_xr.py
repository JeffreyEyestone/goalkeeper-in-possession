import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import pandas as pd,numpy as np,joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.features import matrix
from gkpossession.xr import classifier,fit_calibration,calibrated
from gkpossession.stats import metrics,bootstrap_metrics

def main():
    df=pd.read_parquet(ROOT/'data/processed/actions.parquet');splitfile=ROOT/'config/wc2018_split.json'
    if not splitfile.exists():
        ids=np.array(sorted(json.loads((ROOT/'data/manifests/matches_WC2018.json').read_text())));np.random.default_rng(42).shuffle(ids)
        split={'seed':42,'train':ids[:38].tolist(),'calibration':ids[38:51].tolist(),'test':ids[51:].tolist()};splitfile.write_text(json.dumps(split,indent=2))
    split=json.loads(splitfile.read_text());df['partition']='external'
    for k in ['train','calibration','test']:df.loc[df.match_id.isin(split[k]),'partition']=k
    assert not set(split['train'])&set(split['test']) and not set(split['train'])&set(split['calibration']) and not set(split['test'])&set(split['calibration'])
    out=ROOT/'results/v21';models=ROOT/'data/processed/models';models.mkdir(exist_ok=True)
    rows=[];cvrows=[];selected={};reliability=[];predictions=df.copy();gates=[];splits=[]
    for estimand in ['TF','TA']:
        use=df[df.R1.notna() & (True if estimand=='TF' else df.target_available)].copy()
        train=use[use.partition=='train'];cal=use[use.partition=='calibration'];trX=matrix(train,estimand)
        for partition,g in use.groupby('partition'):
            splits.append({'estimand':estimand,'partition':partition,'n':len(g),'matches':g.match_id.nunique(),'keepers':g.keeper_id.nunique(),'teams':g.team_id.nunique(),'base_rate':g.R1.mean()})
        # Select only within WC2018 training matches. External/test metrics cannot choose a winner.
        family_cv={}
        for family in ['logistic','shallow_tree','hist_gradient']:
            losses=[]
            for fold,(a,b) in enumerate(GroupKFold(4).split(trX,train.R1,train.match_id)):
                model=classifier(family,estimand);model.fit(trX.iloc[a],train.R1.iloc[a].astype(int));p=model.predict_proba(trX.iloc[b])[:,1];losses.append(brier_score_loss(train.R1.iloc[b],p))
            family_cv[family]=float(np.mean(losses));cvrows.append({'estimand':estimand,'family':family,'train_groupcv_brier':float(np.mean(losses))})
        winner=min(family_cv,key=family_cv.get);selected[estimand]={'family':winner,'selection':'lowest 4-fold match-grouped training Brier; no test/external selection','calibration':'isotonic'}
        for family in ['logistic','shallow_tree','hist_gradient']:
            model=classifier(family,estimand);model.fit(trX,train.R1.astype(int));pcal=model.predict_proba(matrix(cal,estimand))[:,1]
            for method in ['isotonic','platt']:
                mapping=fit_calibration(cal.R1.astype(int),pcal,method)
                key=f'{estimand}_{family}_{method}';allrows=df if estimand=='TF' else df[df.target_available]
                pred=calibrated(mapping,model.predict_proba(matrix(allrows,estimand))[:,1],method);predictions.loc[allrows.index,key]=pred
                if family==winner and method=='isotonic':
                    predictions.loc[allrows.index,f'rho_{estimand}']=pred
                    joblib.dump({'model':model,'calibration':mapping,'method':method,'estimand':estimand,'features':list(trX)},models/f'xr_{estimand}.joblib')
                for cohort,g in use[~use.partition.isin(['train','calibration'])].groupby('cohort'):
                    p=predictions.loc[g.index,key];m=metrics(g.R1,p);row={'estimand':estimand,'family':family,'calibration':method,'cohort':cohort,'sample':'test' if cohort=='WC2018' else 'external',**m}
                    if family==winner and method=='isotonic':
                        sample=g.copy();sample['p']=p;ci=bootstrap_metrics(sample,'p')
                        for name,(lo,hi) in ci.items():row[name+'_lo']=lo;row[name+'_hi']=hi
                        bins=np.minimum((np.asarray(p)*10).astype(int),9)
                        for b in range(10):
                            mask=bins==b
                            if mask.any():
                                yy=g.R1.to_numpy()[mask];n=len(yy);q=yy.mean();den=1+1.96**2/n;center=(q+1.96**2/(2*n))/den;rad=1.96*np.sqrt(q*(1-q)/n+1.96**2/(4*n*n))/den
                                reliability.append({'estimand':estimand,'cohort':cohort,'bin':b,'n':n,'mean_prediction':float(np.asarray(p)[mask].mean()),'observed':q,'observed_wilson_lo':center-rad,'observed_wilson_hi':center+rad})
                        material=cohort!='WC2018' and (m['ece']>.10 or abs(m['mean_gap'])>.10 or ci['calibration_slope'][1]<.5 or ci['calibration_slope'][0]>2)
                        if material:gates.append({'gate':'external_calibration','estimand':estimand,'cohort':cohort,'metrics':m,'slope_interval':ci['calibration_slope']})
                        if estimand=='TF' and cohort=='WC2018' and (ci['auc'][1]<.55 or m['prediction_sd']<.025):gates.append({'gate':'A_information_failure',**m})
                    rows.append(row)
                print(key,'done',flush=True)
    predictions.to_parquet(ROOT/'data/processed/xr_predictions.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'xr_metrics.csv',index=False);pd.DataFrame(cvrows).to_csv(out/'xr_model_selection.csv',index=False);pd.DataFrame(splits).to_csv(out/'split_summary.csv',index=False);pd.DataFrame(reliability).to_csv(out/'reliability.csv',index=False)
    (out/'xr_selected_models.json').write_text(json.dumps(selected,indent=2));(out/'scientific_gates.json').write_text(json.dumps(gates,indent=2))
    print('SELECTED',selected,'GATES',gates,flush=True)
if __name__=='__main__':main()
