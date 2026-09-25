import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

def calibration(y,p):
    y=np.asarray(y);z=logit(np.clip(p,1e-5,1-1e-5));X=np.column_stack([np.ones(len(y)),z])
    def loss(b):
        v=X@b;return (np.logaddexp(0,v)-y*v).sum()+1e-8*(b*b).sum()
    fit=minimize(loss,[0.,1.],jac=lambda b:X.T@(expit(X@b)-y)+2e-8*b,method='BFGS')
    b=fit.x;return float(b[0]),float(b[1])

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),0,1)
    if len(np.unique(y))<2:return {}
    bins=np.minimum((p*10).astype(int),9);ece=sum(abs(p[bins==b].mean()-y[bins==b].mean())*(bins==b).mean() for b in range(10) if (bins==b).any())
    intercept,slope=calibration(y,p);base=y.mean();brier=brier_score_loss(y,p)
    return {'n':len(y),'base_rate':float(base),'mean_prediction':float(p.mean()),'mean_gap':float(p.mean()-base),'prediction_sd':float(p.std()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'brier':float(brier),'brier_skill_cohort_constant':float(1-brier/(base*(1-base))),'log_loss':float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),'ece':float(ece),'calibration_intercept':intercept,'calibration_slope':slope}

def cluster_bootstrap_indices(df, seed=42):
    """Return row indices for one replacement cluster draw, preserving multiplicity."""
    rng=np.random.default_rng(seed);groups=[g.index.to_numpy() for _,g in df.groupby('match_id')]
    return np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])

def bootstrap_metrics(df,pcol,label='R1',reps=500,seed=42):
    rng=np.random.default_rng(seed);groups=[g.index.to_numpy() for _,g in df.groupby('match_id')];draws=[]
    for _ in range(reps):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);g=df.loc[ix];m=metrics(g[label],g[pcol])
        if m:draws.append(m)
    return {k:(float(np.quantile([r[k] for r in draws],.025)),float(np.quantile([r[k] for r in draws],.975))) for k in draws[0] if k!='n'}
