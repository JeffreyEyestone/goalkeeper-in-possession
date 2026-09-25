import numpy as np
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from .features import TF_NUM,TA_NUM,TF_CAT,matrix

def classifier(family,estimand):
    nums=TF_NUM if estimand=='TF' else TA_NUM
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),TF_CAT)],sparse_threshold=0)
    model={'logistic':LogisticRegression(C=1,max_iter=1000,random_state=42),'shallow_tree':DecisionTreeClassifier(max_depth=3,min_samples_leaf=50,random_state=42),'hist_gradient':HistGradientBoostingClassifier(max_iter=300,max_depth=5,learning_rate=.05,min_samples_leaf=20,early_stopping=False,random_state=42)}[family]
    return Pipeline([('pre',pre),('model',model)])

def fit_calibration(y,p,method):
    if method=='isotonic':return IsotonicRegression(out_of_bounds='clip').fit(p,y)
    return LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(p,1e-5,1-1e-5)).reshape(-1,1),y)

def calibrated(cal,p,method):
    if method=='isotonic':return cal.predict(p)
    return cal.predict_proba(logit(np.clip(p,1e-5,1-1e-5)).reshape(-1,1))[:,1]
