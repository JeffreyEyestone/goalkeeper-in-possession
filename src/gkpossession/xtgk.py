"""Expected value: scoring never reads outcome or resolution coordinates."""
import numpy as np
from .features import matrix
from .surfaces import lookup

def score(frame,rho,success_model,failure_model,surface,estimand='TF',kappa=1,pressure=True):
    X=matrix(frame,estimand);S=np.maximum(success_model.predict(X),0);C=np.maximum(failure_model.predict(X),0)
    V=lookup(surface,X.origin_x,X.origin_y,pressure=X.origin_pressure if pressure else None)
    return {'success_payoff':S,'failure_cost':C,'origin_value':V,'value':np.asarray(rho)*S-V-(1-np.asarray(rho))*kappa*C}

def score_components(rho, success_payoff, origin_value, failure_cost, kappa=1):
    """Canonical xT-GK value for already-computed action components."""
    return np.asarray(rho)*np.asarray(success_payoff) - np.asarray(origin_value) - (1-np.asarray(rho))*kappa*np.asarray(failure_cost)
