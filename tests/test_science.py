import json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.features import matrix,TF_NUM,TF_CAT,distance
from gkpossession.labels import study_labels

def ev(i,t,team=1,typ='Pass',**kw):
    return dict(id=str(i),index=i,period=1,timestamp=f'00:00:{t:06.3f}',type={'name':typ},team={'id':team},possession_team={'id':team},location=[20,40],pass_={},**kw)

def event(i,t,team=1,typ='Pass',**kw):
    e=ev(i,t,team,typ,**kw);e.pop('pass_');
    if typ=='Pass':e.setdefault('pass',{})
    return e

def test_r1_contested_opponent_control():
    events=[event(1,0),event(2,2,typ='Ball Receipt*',ball_receipt={'outcome':{'name':'Incomplete'}}),event(3,2,typ='Duel'),event(4,2,team=2),event(5,20,team=2)]
    lab=study_labels(events,0);assert lab['R1']['label']==0 and lab['R1']['state_id']=='4'

def test_r1_completed_receipt_then_turnover_differs_horizon():
    events=[event(1,0),event(2,1,typ='Ball Receipt*'),event(3,3,team=2,typ='Carry'),event(4,20,team=2)]
    lab=study_labels(events,0);assert lab['R1']['label']==1 and lab['R2']['label']==0

def test_own_shot_before_horizon():
    events=[event(1,0),event(2,1,typ='Shot'),event(3,3,team=2,typ='Carry'),event(4,20,team=2)]
    assert study_labels(events,0)['R2']['label']==1

def test_dead_ball_not_silent_retention():
    a=event(1,0);a['pass']={'outcome':{'name':'Out'}}
    assert study_labels([a,event(2,20)],0)['R1']['status']=='dead_ball'

def test_empty_horizon_is_not_success_except_exact_legacy():
    a=event(1,0);lab=study_labels([a],0)
    assert lab['R2']['label'] is None and lab['R3']['label']==1

def test_no_period_crossing():
    a=event(1,0);b=event(2,1,typ='Carry');b['period']=2
    assert study_labels([a,b],0)['R1']['label'] is None

def test_tf_outcome_invariance_and_forbidden_columns():
    row={k:1. for k in TF_NUM};row.update({k:'known' for k in TF_CAT});df=pd.DataFrame([row]);base=matrix(df)
    for outcome in ['Complete','Out','Incomplete',None]:
        altered=df.assign(outcome=outcome,end_x=-999,end_y=999,receiver_pressure=True,target_available=False,realized_distance_m=100)
        pd.testing.assert_frame_equal(matrix(altered),base)
    for forbidden in ['height','technique','play_pattern','end_x','target_x','receiver_pressure','R1','complete','native_distance','target_available']:
        with pytest.raises(ValueError):matrix(df,columns=TF_NUM+[forbidden])

def test_standardized_mapping_is_anisotropic():
    assert distance(120,0)==105 and distance(0,80)==68
    assert distance(120,80)==pytest.approx(np.hypot(105,68))

def test_ta_cannot_impute_missing_target():
    with pytest.raises(ValueError):matrix(pd.DataFrame({'target_x':[np.nan],'target_y':[20]}),'TA')

def test_opponent_control_precedes_their_pass_going_out():
    a=event(1,0);b=event(2,2,team=2);b['pass']={'outcome':{'name':'Out'}}
    assert study_labels([a,b],0)['R1']['label']==0

def test_frozen_match_split_is_exhaustive_and_disjoint():
    split=json.loads((ROOT/'config/wc2018_split.json').read_text());sets=[set(split[k]) for k in ['train','calibration','test']]
    assert [len(x) for x in sets]==[38,13,13]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert set.union(*sets)==set(json.loads((ROOT/'data/manifests/matches_WC2018.json').read_text()))

def test_restart_rights_do_not_depend_on_restart_outcome():
    a=event(1,0);a['pass']={'outcome':{'name':'Out'}}
    b=event(2,25,team=2);b['pass']={'type':{'name':'Throw-in'}}
    for outcome in [None,'Complete','Incomplete','Out']:
        if outcome:b['pass']['outcome']={'name':outcome}
        lab=study_labels([a,b],0)['R1'];assert lab['label']==0 and lab['status']=='restart_rights'

def test_probability_metrics_include_one_in_ece():
    from gkpossession.stats import metrics
    m=metrics([1,0,1,0],[1.,1.,0.,0.]);assert m['ece']==pytest.approx(.5)

def test_calibration_intercept_slope_are_logistic_parameters():
    from gkpossession.stats import calibration
    rng=np.random.default_rng(42); p=np.array([.1,.2,.4,.7,.9]*1000); y=rng.binomial(1,p)
    a,b=calibration(y,p); assert abs(a)<.15 and abs(b-1)<.2
    a2,b2=calibration(y, np.clip(p*.7+.15,1e-5,1-1e-5)); assert b2>0 and abs(b2-1)>.2

def test_cluster_bootstrap_preserves_repeated_draws():
    from gkpossession.stats import cluster_bootstrap_indices
    d=pd.DataFrame({'match_id':[1,2], 'v':[1.,3.]})
    draws=[cluster_bootstrap_indices(d,seed=i) for i in range(20)]
    assert any(len(set(ix))<2 for ix in draws)

def test_pressure_is_binary_free_or_pressured():
    d=pd.DataFrame({'origin_pressure':[0,1,0,1]})
    assert set(np.where(d.origin_pressure==1,'PRESSURED','FREE'))=={'FREE','PRESSURED'}

def test_context_fit_support_invariants():
    d=pd.read_csv(ROOT/'results/v21/contextual_fit_support.csv')
    assert d.common_support_pct.between(0,1).all()
    assert d.unsupported_state_mass.between(0,1).all()
    assert np.allclose(d.common_support_pct+d.unsupported_state_mass,1)
    assert (d.max_weight_proxy>=1).all()

def test_no_arbitrary_context_adjustment_and_distinct_xr_xt():
    src=(ROOT/'scripts/context_evidence_gate.py').read_text()
    assert '*0.01' not in src and '* 0.01' not in src
    d=pd.read_csv(ROOT/'results/v21/keeper_context_rank_changes.csv')
    assert {'xr_gk_plus','context_adjusted_xr_gk_plus','xt_gk'} <= set(d.columns)

def test_ta_geometry_is_separate_from_tf_features():
    from gkpossession.features import TA_NUM
    assert not set(TF_NUM)&{'target_x','target_y','target_distance_m','target_angle'}
    assert set(TA_NUM)-set(TF_NUM)=={'target_x','target_y','target_distance_m','target_angle'}

def test_receiver_extension_is_ta_only_and_metrics_are_distinct():
    d=pd.read_csv(ROOT/'results/v21_r2/receiver_model_comparison.csv')
    assert set(d.model)=={'TA-GEO','TA-ROLE','TA-PLAYER'}
    assert d.n.nunique()==1 and (d.receivers>0).all()

def test_intercept_only_root_solves_likelihood_and_differs_from_slope():
    import importlib.util
    spec=importlib.util.spec_from_file_location('r32a_finalize',ROOT/'scripts/r32a_finalize.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); intercept_only=mod.intercept_only
    from scipy.special import expit, logit
    rng=np.random.default_rng(7); p=np.linspace(.05,.95,1000); y=rng.binomial(1,expit(logit(p)+.4))
    a=intercept_only(y,p); assert abs(a-.4)<.15
    # A slope distortion cannot be represented exactly by an intercept shift.
    y2=rng.binomial(1,expit(1.6*logit(p))); a2=intercept_only(y2,p); assert abs(a2)<.2

@pytest.mark.parametrize('fixture',json.loads((ROOT/'tests/fixtures/inspected_restart_sequences.json').read_text()))
def test_inspected_real_restart_sequences(fixture):
    result=study_labels(fixture['events'],0)['R1']
    assert result['label']==fixture['expected_label']
    assert result['state_id']==fixture['expected_state_id']
    assert result['status']=='restart_rights'
