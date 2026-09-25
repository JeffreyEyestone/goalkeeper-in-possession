"""Outcome-invariant expected branch payoff regression on approved features."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .features import matrix,TF_NUM,TA_NUM,TF_CAT

class ShrunkBranchMean:
    def __init__(self,estimand='TF',k=30):self.estimand=estimand;self.k=k
    def keys(self,frame):
        x=matrix(frame,self.estimand)
        origin=(x.origin_x//20).astype(int).astype(str)+'/'+(x.origin_y//20).astype(int).astype(str)
        key=x.goal_kick.astype(str)+'/'+x.body_part.astype(str)+'/'+origin
        if self.estimand=='TA':key=key+'/'+(x.target_x//20).astype(int).astype(str)+'/'+(x.target_y//20).astype(int).astype(str)
        return key
    def fit(self,frame,y):
        keys=self.keys(frame);y=np.asarray(y,dtype=float);self.global_mean=float(y.mean());self.means={}
        for key in keys.unique():
            v=y[(keys==key).to_numpy()];self.means[key]=(v.sum()+self.k*self.global_mean)/(len(v)+self.k)
        return self
    def predict(self,frame):return self.keys(frame).map(self.means).fillna(self.global_mean).to_numpy()

def regressor(estimand):
    nums=TF_NUM if estimand=='TF' else TA_NUM
    pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),TF_CAT)],sparse_threshold=0)
    return Pipeline([('pre',pre),('reg',HistGradientBoostingRegressor(max_iter=100,max_depth=2,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42))])
