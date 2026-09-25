"""Explicit whitelist; diagnostic/outcome columns can never enter model matrices."""
import numpy as np
import pandas as pd
TF_NUM=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference']
TF_CAT=['body_part']
TA_NUM=TF_NUM+['target_x','target_y','target_distance_m','target_angle']
FORBIDDEN={'height','technique','play_pattern','end_x','end_y','realized_distance_m','native_distance','outcome','complete','target_available','receiver_pressure','R1','R2','R3','loss_x','loss_y','retain','length','angle','fwd','lat'}

def matrix(frame,estimand='TF',columns=None):
    allowed=(TF_NUM if estimand=='TF' else TA_NUM)+TF_CAT
    if estimand not in ('TF','TA'):raise ValueError('Unknown estimand')
    requested=allowed if columns is None else list(columns)
    if set(requested)-set(allowed):raise ValueError('Forbidden or unregistered feature')
    if estimand=='TA' and frame[['target_x','target_y']].isna().any().any():raise ValueError('TA requires a defensible target; imputation forbidden')
    return frame[requested].copy()

def distance(dx,dy):return np.hypot(np.asarray(dx)*105/120,np.asarray(dy)*68/80)
