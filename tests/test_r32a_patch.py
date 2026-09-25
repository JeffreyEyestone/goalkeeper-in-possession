import sys, numpy as np
from pathlib import Path
from scipy.special import expit, logit
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gkpossession.calibration import intercept_only

def test_canonical_intercept_only_estimator():
    rng=np.random.default_rng(4); p=np.linspace(.05,.95,10000); y=rng.binomial(1,p); assert abs(intercept_only(y,p)) < .08
    y=rng.binomial(1,expit(logit(p)+.55)); assert abs(intercept_only(y,p)-.55)<.08
    y=rng.binomial(1,expit(1.8*logit(p))); a=intercept_only(y,p); assert abs(a)<.08
