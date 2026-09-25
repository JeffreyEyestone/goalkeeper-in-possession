import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.xtgk import score
from gkpossession.failure_cost import ShrunkBranchMean
class M:
    def predict(self,X): return np.full(len(X),.01)
def row():
    return pd.DataFrame([{'origin_x':10.,'origin_y':40.,'origin_pressure':0,'goal_kick':1,'time_minutes':3.,'second_half':0,'score_difference':0,'body_part':'Left Foot','target_available':True,'target_x':50.,'target_y':40.,'target_distance_m':35.,'target_angle':0.,'end_x':11.,'end_y':41.,'outcome':'Complete','R1_state_x':80.,'R1_state_y':40.,'observed_success_payoff':.02,'observed_failure_cost':.03}])
def surf():
    return {'grid':(16,12),'V':np.full(192,.01),'Vp':np.full((192,2),.01),'T':np.full(192,.02)}
def test_expected_value_ignores_realized_outcome_geometry():
    a=row();v=score(a,np.array([.7]),M(),M(),surf())['value'];b=a.assign(end_x=119,end_y=79,outcome='Out',R1_state_x=1,R1_state_y=1,observed_failure_cost=.99);w=score(b,np.array([.7]),M(),M(),surf())['value'];np.testing.assert_allclose(v,w)
def test_failure_cost_model_only_uses_whitelisted_features():
    a=row().assign(target_available=False);m=ShrunkBranchMean('TF').fit(a,[.02]);p=m.predict(a.assign(end_x=119,outcome='Out',R1_state_x=1));assert p[0]>=0
