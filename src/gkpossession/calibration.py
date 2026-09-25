"""Canonical probability calibration estimators."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

def intercept_only(y, p):
    """Binomial MLE alpha for expit(logit(p) + alpha), beta fixed at one."""
    y = np.asarray(y, dtype=float)
    z = logit(np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5))
    return float(brentq(lambda a: np.sum(y - expit(z + a)), -30.0, 30.0))
