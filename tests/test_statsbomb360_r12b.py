"""Numerical and specification checks for the locked R1.2B attribution."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_statsbomb360_r12b_attribution import (
    OPTIONS, polygon_area_fraction, spec_for_scope, weighted_auc_tie_aware,
)


def test_tie_aware_auc_equals_repeated_rows_and_sklearn():
    y = np.array([0,1,1,0,0,1,1,0])
    p = np.array([.1,.2,.2,.2,.7,.7,.9,.9])
    multiplicity = np.array([3,2,1,4,2,1,2,1])
    actual = weighted_auc_tie_aware(y,p,multiplicity[:,None])[0]
    assert np.isclose(actual,roc_auc_score(y,p,sample_weight=multiplicity),atol=1e-12)
    assert np.isclose(actual,roc_auc_score(np.repeat(y,multiplicity),np.repeat(p,multiplicity)),atol=1e-12)


def test_primary_nested_spec_and_constant_competition():
    for scope in ('ALL','WC2022','EURO2024'):
        spec=spec_for_scope(scope)
        assert set(spec['M3'])-set(spec['M2'])==set(OPTIONS)
        assert len(spec['M3'])-len(spec['M2'])==2
        assert ('comp_euro' in spec['M2'])==(scope=='ALL')
        assert set(spec['M3C'])-set(spec['M2C'])==set(OPTIONS)
        assert set(spec['M4'])-set(spec['M3'])=={'receiver_option_count'}


def test_visible_area_shoelace_uses_standardized_meters():
    polygon=json.dumps({'visible_area':[0,0,120,0,120,80,0,80]})
    area,fraction=polygon_area_fraction(polygon)
    assert np.isclose(area,7140)
    assert np.isclose(fraction,1)
