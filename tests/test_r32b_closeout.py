from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_closeout_roles_and_curves_are_populated():
 r=ROOT/'results/r32b_closeout'; m=pd.read_csv(r/'receiver_role_mapping.csv'); assert m.receiver_role.ne('OTHER_UNKNOWN').mean()>.8; assert (m.receiver_role=='DM_6').sum()>0; assert (m.receiver_role=='STRIKER').sum()>0
 c=pd.read_csv(r/'restart_curve.csv'); assert c.R1_prediction.notna().all() and c.value_prediction.notna().all()
