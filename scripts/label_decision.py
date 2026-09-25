from pathlib import Path
import json,sys
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.features import TF_NUM,TF_CAT,TA_NUM
x=pd.read_parquet(ROOT/'data/processed/actions.parquet')
summary=pd.read_csv(ROOT/'results/v21/retention_label_study.csv').query("dimension=='all'")
lines=['# Retention label decision before model fitting','','Primary: **R1 resolution retention**, meaning own-team first controlled action/receipt or an explicitly annotated awarded restart at distribution resolution. This is team retention/possession rights, not pure pass execution. Strict live-ball control (`R1_control`), five-second control (`R2`), exact legacy ten-second (`R3`), ten-second controlled (`R3_control`) and fifteen-second control (`R15`) are retained sensitivities. No keeper rankings or predictive model results were consulted.', '',
'Live-ball resolution scans up to 15 seconds within the same period. Successful Ball Receipt*, Carry, deliberate Pass (excluding No Touch), Shot, successful recovery/dribble/interception establish control; unresolved duel/50-50/clearance alone do not. A deliberate pass establishes control at its origin even if that subsequent pass later fails. This convention includes one-touch deliberate redirections; contested headers without a coded controlled action remain unresolved.', '',
'Dead-ball extension uses the first explicitly marked restart/penalty within 120 seconds and the same period, provided no intervening controlled action occurs. The restart origin is the payoff state. It never assumes all out-of-play passes are failures, never reads restart outcome to determine rights, and never carries labels across halves. Missing/unresolvable cases remain excluded from supervision, with predictions on those cases interpreted as extrapolations. “All-action TF” means no target-availability selection, not that censored labels are magically observed.', '',
'R1 is selected because the probability event and modeled branch-state payoff can refer to the **same resolution event**. R2 requires a proxy for control between events and excludes more contested cases. Legacy R3 labels empty windows retained and uses possession metadata rather than observed control, and can blend intervening loss/recovery; it offers near-complete labels at the cost of a less defensible immediate payoff. Those weaknesses outweigh its historical familiarity. R1 branches will be modeled conditionally using legitimate release features; actual resolution states are training targets only.', '',
'| Cohort | Label | Positive | Negative | Ambiguous | Dead ball unresolved | Censored | Base rate |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary.itertuples():lines.append(f'| {r.cohort} | {r.label} | {r.positive} | {r.negative} | {r.ambiguous} | {r.dead_ball} | {r.censored} | {r.base_rate:.4f} |')
lines+=['','The cohort/band/restart tables are in retention_label_study.csv. Distance strata use standardized **realized** displacement for measurement diagnostics, not TF predictors. Expanded R1 restart-rights labels are a documented measurement convention; repeat analysis under strict R1 to reveal its effect. Label quality still depends on event coding; no claim of video-validated ground truth is made.']
(ROOT/'reports/RETENTION_LABEL_DECISION.md').write_text('\n'.join(lines)+'\n')
selection=[]
for c,g in x.groupby('cohort'):
    tab=pd.crosstab(g.complete,g.target_available).reindex(index=[0,1],columns=[False,True],fill_value=0)
    a,b=tab.loc[1,True],tab.loc[1,False];d,e=tab.loc[0,True],tab.loc[0,False]
    logor=np.log((a+.5)*(e+.5)/((b+.5)*(d+.5)));se=np.sqrt(sum(1/(v+.5) for v in [a,b,d,e]))
    selection.append({'cohort':c,'completed_availability':a/(a+b),'noncompleted_availability':d/(d+e),'availability_risk_difference':a/(a+b)-d/(d+e),'odds_ratio_haldane':np.exp(logor),'or_lo':np.exp(logor-1.96*se),'or_hi':np.exp(logor+1.96*se),'n':len(g)})
pd.DataFrame(selection).to_csv(ROOT/'results/v21/target_selection_outcome.csv',index=False)
text=['# Intended-target missingness and selection','','Target availability is an outcome-dependent observation mechanism. No target imputation or missing-at-random assumption is made. Complete-case TA estimates a selected subset; TF is scored on that same subset for an apples-to-apples comparison, in addition to the full eligible population.', '', '| Cohort | n | Complete availability | Non-complete availability | Difference | OR |','|---|---:|---:|---:|---:|---:|']
for r in selection:text.append(f"| {r['cohort']} | {r['n']} | {r['completed_availability']:.2%} | {r['noncompleted_availability']:.2%} | {r['availability_risk_difference']:.2%} | {r['odds_ratio_haldane']:.1f} |")
text+=['','Odds ratios use a 0.5-cell correction because some completed-pass strata have no missing targets. Intervals in the CSV are simple contingency-table descriptive intervals, not causal effects or match-clustered inference. Strong outcome dependence directly refutes treating the selected subset as random.', '',
'`target_selection_strata.csv` covers cohort × completion/outcome, restart, origin pressure, height, body part, technique, play pattern, keeper, team, match, half, origin bands, time bands, score state, combined action family and era. Cohort determines era/population; era and competition effects cannot be separated in this design. Small-n keeper/team/match rates are descriptive only. A multivariable missingness model would not identify the targets that were never observed.', '',
'Height, technique and play pattern are analyzed for missingness but withheld from the TF matrix. No target-availability flag, pass outcome, realized geometry or downstream pressure can enter TF. TA target location is a retrospective intended-receipt annotation; its collection mechanism can still leak subtler outcome information, requiring target-coordinate coarsening/precision sensitivity.']
(ROOT/'reports/TARGET_MISSINGNESS_AND_SELECTION.md').write_text('\n'.join(text)+'\n')

# Additional selection strata from available release context; these are never model features.
extra=x.copy()
extra['origin_x_band']=pd.cut(extra.origin_x,[-float('inf'),6,18,40,80,float('inf')]).astype(str)
extra['origin_y_band']=pd.cut(extra.origin_y,[-float('inf'),20,60,float('inf')]).astype(str)
extra['time_band']=pd.cut(extra.time_minutes,[-float('inf'),15,30,45,60,float('inf')]).astype(str)
extra['score_state']=np.sign(extra.score_difference)
extra['action_family']=extra.goal_kick.astype(str)+' / '+extra.height+' / '+extra.body_part
extra['era']=extra.cohort.map({'WC2018':'2018','WC2022':'2022','T2024':'2024','PL1516':'2015-16','WSL2021':'2020-21'})
newdims=['R1','origin_x_band','origin_y_band','time_band','score_state','action_family','era']
additional=[]
for dim in newdims:
    for (c,lev),g in extra.groupby(['cohort',dim],dropna=False,observed=True):
        additional.append({'cohort':c,'dimension':dim,'level':str(lev),'n':len(g),'target_available':int(g.target_available.sum()),'availability_rate':g.target_available.mean(),'completion_rate':g.complete.mean(),'R1_rate':g.R1.mean()})
pth=ROOT/'results/v21/target_selection_strata.csv';old=pd.read_csv(pth);old=old[~old.dimension.isin(newdims)];pd.concat([old,pd.DataFrame(additional)],ignore_index=True).to_csv(pth,index=False)

registry=[]
for f in TF_NUM+TF_CAT:
    registry.append({'feature':f,'estimands':['TF','TA'],'timestamp':'release / observable action family','leakage_status':'allowed_with_annotation_caveat','derivation':'prior-score accumulation' if f=='score_difference' else 'current action annotation','missingness_by_cohort':x.groupby('cohort')[f].apply(lambda s:float(s.isna().mean())).to_dict()})
for f in [f for f in TA_NUM if f not in TF_NUM]:registry.append({'feature':f,'estimands':['TA'],'timestamp':'retrospective intent annotation','leakage_status':'secondary_only; selected sample and collection precision caveat','derivation':'unique same-team, same-period UUID-linked intended receipt; never endpoint fallback','missingness_by_cohort':x.groupby('cohort')[f].apply(lambda s:float(s.isna().mean())).to_dict()})
(ROOT/'config/feature_registry.json').write_text(json.dumps(registry,indent=2))
config=json.loads((ROOT/'config/analysis.json').read_text());config['primary_label']='R1';config['primary_label_definition']='own control or explicitly awarded restart at immediate resolution';(ROOT/'config/analysis.json').write_text(json.dumps(config,indent=2))
print(summary.to_string(index=False))
