import numpy as np
from gkpossession.xtgk import score_components

def test_canonical_closeout_formula_and_old_formula_differs():
    rho=np.array([.4,.8]); s=np.array([2.,3.]); v=np.array([.7,.2]); c=np.array([.5,.4])
    got=score_components(rho,s,v,c); expected=rho*s-v-(1-rho)*c
    old=rho*(s-v)-(1-rho)*c
    assert np.allclose(got,expected); assert not np.allclose(got,old)
