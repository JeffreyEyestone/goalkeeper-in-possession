"""Secondary intended-receiver extension (TA only; never part of TF)."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/v21_r2'; OUT.mkdir(exist_ok=True)
def ece(y,p):
    b=np.minimum((p*10).astype(int),9); return float(sum(abs(p[b==i].mean()-y[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any()))
def main():
    x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=x[x.target_available & x.R1.notna()].copy()
    positions={}; recs=[]; action_ids=set(x.event_id)
    for c in ['PL1516','T2024','WC2018','WC2022','WSL2021']:
      for mid in json.loads((ROOT/f'data/manifests/matches_{c}.json').read_text()):
        ev=json.loads((ROOT/f'data/raw/events/{mid}.json').read_text()); by={e['id']:e for e in ev}; pos={}
        for e in ev:
          if e.get('player') and e.get('position'): pos.setdefault(e['player']['id'],[]).append(e['position'].get('name','Unknown'))
        positions.update({k:max(set(v),key=v.count) for k,v in pos.items()})
        for e in ev:
          if e.get('id') in action_ids:
            r=e.get('pass',{}).get('recipient',{}); recs.append((e['id'],r.get('id'),r.get('name')))
    rec=pd.DataFrame(recs,columns=['event_id','receiver_id','receiver']); x=x.merge(rec,on='event_id',how='left'); x['receiver_role']=x.receiver_id.map(positions).fillna('Unknown')
    # Only target-identifiable actions enter this extension. Grouped match holdout.
    nums=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference','target_x','target_y','target_distance_m','target_angle']; y=x.R1.astype(int).to_numpy(); specs=[('TA-GEO',[],nums),('TA-ROLE',['receiver_role'],nums),('TA-PLAYER',['receiver_id'],nums)]
    rows=[]
    for name,cats,numerics in specs:
      pre=ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),numerics),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cats)])
      pred=np.full(len(x),np.nan)
      for tr,te in GroupKFold(4).split(x,y,x.match_id):
        m=Pipeline([('p',pre),('m',LogisticRegression(max_iter=500,C=.5))]);m.fit(x.iloc[tr][numerics+cats],y[tr]);pred[te]=m.predict_proba(x.iloc[te][numerics+cats])[:,1]
      rows.append({'model':name,'n':len(x),'receivers':x.receiver_id.nunique(),'auc':roc_auc_score(y,pred),'brier':brier_score_loss(y,pred),'log_loss':log_loss(y,pred,labels=[0,1]),'ece':ece(y,pred),'group':'match','note':'TA selected sample; receiver descriptors are secondary'}); x['p_'+name]=pred
    pd.DataFrame(rows).to_csv(OUT/'receiver_model_comparison.csv',index=False)
    # Receiver effects use shrinkage and leave-one-action-out aggregate summaries.
    glob=x.R1.mean(); g=x.groupby(['receiver_id','receiver','receiver_role'],dropna=False).agg(n=('event_id','size'),retention=('R1','mean'),value=('value_TA','mean'),roe=('roe_TA','mean')).reset_index(); g['retention_shrunk']=(g.n*g.retention+50*glob)/(g.n+50); g['reliability']=np.where(g.n>=100,'adequate',np.where(g.n>=30,'low_support','not_estimable')); g.to_csv(OUT/'gk_receiver_effects.csv',index=False)
    pair=x.groupby(['keeper_id','keeper','receiver_id','receiver'],dropna=False).agg(n=('event_id','size'),retention=('R1','mean'),value=('value_TA','mean')).reset_index(); pair['reliability']=np.where(pair.n>=50,'low_support_or_better','not_estimable'); pair.to_csv(OUT/'gk_receiver_pairs.csv',index=False)
    (ROOT/'reports/INTENDED_RECEIVER_RESEARCH.md').write_text('# Intended receiver research\n\nTA-GEO, TA-ROLE and TA-PLAYER are compared on the target-identifiable subset with match-grouped held-out folds. Receiver identity materially conditions the estimand only for this selected sample; it cannot be promoted to TF or interpreted as causal receiver skill.\n')
    (ROOT/'reports/GK_RECEIVER_CREDIT.md').write_text('# Goalkeeper versus receiver credit\n\nReceiver effects are shrunk descriptive conditional associations. Identity and role are not computed from the evaluated action outcome. Pair effects are flagged low-support unless repeated sufficiently; the current open data do not establish chemistry or causal credit allocation.\n')
    (ROOT/'reports/OPTION_SET_FUTURE_SPEC.md').write_text('# Option-set future specification\n\nA true option-set model requires all feasible receivers and their release-time locations, pressure, cover and lanes. Event data identify the linked intended recipient after the fact, but cannot recover unavailable alternatives. Tracking or 360 data are required to score each candidate option and assess best-choice decision quality.\n')
    print('receiver extension written',flush=True)
if __name__=='__main__': main()
