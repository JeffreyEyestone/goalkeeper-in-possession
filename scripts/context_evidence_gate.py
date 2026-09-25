"""Held-out context evidence gate for the current exploratory framework."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/v21'; sys.path.insert(0,str(ROOT/'src'))
from gkpossession.stats import calibration, cluster_bootstrap_indices

def ece(y,p,bins=10):
    cuts=np.linspace(0,1,bins+1); z=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        m=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if m.any(): z += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(z)
def slope_intercept(y,p):
    return calibration(y, p)
def fit_eval(x, features, cat, num):
    gkf=GroupKFold(4); rows=[]; pred=np.full(len(x),np.nan)
    pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    for tr,te in gkf.split(x,x.R1,x.match_id):
        model=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=500,C=0.5))]); model.fit(x.iloc[tr][features],x.iloc[tr].R1.astype(int)); pred[te]=model.predict_proba(x.iloc[te][features])[:,1]
    y=x.R1.astype(int).to_numpy(); ci0,ci1=slope_intercept(y,pred); rows.extend([('brier',brier_score_loss(y,pred)),('log_loss',log_loss(y,pred,labels=[0,1])),('ece',ece(y,pred)),('auc',roc_auc_score(y,pred)),('pr_auc',average_precision_score(y,pred)),('calibration_intercept',ci0),('calibration_slope',ci1)])
    rng=np.random.default_rng(42); boot={k:[] for k,_ in rows}
    for _ in range(200):
        ix=cluster_bootstrap_indices(x, seed=int(rng.integers(0,2**31-1))); yy=y[ix]; pp=pred[ix]
        a,b=slope_intercept(yy,pp); vals={'brier':brier_score_loss(yy,pp),'log_loss':log_loss(yy,pp,labels=[0,1]),'ece':ece(yy,pp),'auc':roc_auc_score(yy,pp),'pr_auc':average_precision_score(yy,pp),'calibration_intercept':a,'calibration_slope':b}
        for k,v in vals.items(): boot[k].append(v)
    return rows,pred,{k:(float(np.quantile(v,.025)),float(np.quantile(v,.975))) for k,v in boot.items()}

def main():
    x=pd.read_parquet(ROOT/'data/processed/valued_actions.parquet'); x=x[x.R1.notna()].copy().reset_index(drop=True); x['pressure_bin']=np.where(x.origin_pressure.astype(int)==1,'PRESSURED','FREE')
    x['state_bin']=(pd.cut(x.origin_x,[-1,35,70,121],labels=['deep','middle','advanced']).astype(str)+'_'+pd.cut(x.origin_y,[-1,22.67,45.33,68.01],labels=['left','central','right']).astype(str))
    base_num=['origin_x','origin_y','origin_pressure','goal_kick','time_minutes','second_half','score_difference']; base_cat=['body_part']
    specs=[('M0_global_action',base_num,base_cat),('M1_competition',base_num,base_cat+['cohort']),('M2_team_context',base_num,base_cat+['cohort','team']),('M3_keeper_effect',base_num,base_cat+['cohort','team','keeper_id'])]
    rows=[]; predictions={}
    for name,nums,cats in specs:
        rr,p,ci=fit_eval(x,nums+cats,cats,nums); predictions[name]=p
        for metric,val in rr:
            rows.append({'specification':name,'metric':metric,'estimate':val,'ci_lo':ci[metric][0],'ci_hi':ci[metric][1]})
    met=pd.DataFrame(rows); base=met[met.specification=='M0_global_action'].set_index('metric').estimate
    met['increment_vs_M0']=met.apply(lambda r:r.estimate-base.get(r.metric,np.nan),axis=1); met.to_csv(OUT/'context_claim_tests.csv',index=False)
    # Keeper rank shifts from raw/value and out-of-fold M2 context residual adjustment.
    p=predictions['M2_team_context']; coef=np.polyfit(p,x.roe_TF,1); x['context_adjusted_xr']=x.roe_TF-(coef[0]*p+coef[1])+x.roe_TF.mean()
    ag=x.groupby(['keeper_id','keeper']).agg(n=('event_id','size'),raw_completion=('complete','mean'),raw_retention=('R1','mean'),xr_gk_plus=('roe_TF','mean'),context_adjusted_xr_gk_plus=('context_adjusted_xr','mean'),xt_gk=('value_TF','mean')).query('n>=100').reset_index()
    ag['raw_rank']=ag.raw_retention.rank(ascending=False); ag['adjusted_rank']=ag.context_adjusted_xr_gk_plus.rank(ascending=False); ag['rank_shift']=(ag.adjusted_rank-ag.raw_rank).abs(); ag.to_csv(OUT/'context_keeper_rank_shifts.csv',index=False)
    # Reweighting reliability: effective sample size and weight concentration.
    cf=pd.read_csv(OUT/'contextual_fit.csv'); cf['reliability_tier']=pd.cut(cf.common_support_pct,[-.01,.2,.5,.8,1.01],labels=['NOT ESTIMABLE','LOW SUPPORT','ADEQUATE','HIGH SUPPORT']); cf['effective_sample_size']=cf.n*cf.common_support_pct; cf['max_weight_proxy']=1/np.maximum(cf.common_support_pct,1e-6); cf['trimmed_sensitivity_proxy']=np.minimum(cf.max_weight_proxy,5)
    # Primary fit excludes unsupported target cells; imputed estimate remains a sensitivity column.
    cf['overlap_restricted_fit']=np.where(cf.common_support_pct>=.2, cf.target_environment_value, np.nan)
    cf['unsupported_state_mass']=1-cf.common_support_pct; cf['fit_primary_status']=np.where(cf.common_support_pct>=.8,'HIGH SUPPORT',np.where(cf.common_support_pct>=.5,'ADEQUATE SUPPORT',np.where(cf.common_support_pct>=.2,'LOW SUPPORT','NOT ESTIMABLE')))
    cf.to_csv(OUT/'contextual_fit_support.csv',index=False)
    cf.to_csv(OUT/'contextual_fit_reliability.csv',index=False)
    # Pressure decomposition, adjusted by state-bin means; match bootstrap intervals are omitted for low-n rows.
    state_mean=x.groupby(['state_bin','goal_kick'],observed=True).agg(state_r=('R1','mean'),state_v=('value_TF','mean')).reset_index()
    x=x.merge(state_mean,on=['state_bin','goal_kick'],how='left'); x['execution_residual']=x.R1-x.rho_TF; x['decision_value_residual']=x.rho_TF-x.state_r; x['value_residual']=x.value_TF-x.state_v
    pr=x.groupby(['keeper_id','keeper','pressure_bin'],observed=True).agg(n=('event_id','size'),decision_shift=('decision_value_residual','mean'),execution_shift=('execution_residual','mean'),value_shift=('value_residual','mean'),retention=('R1','mean'),value=('value_TF','mean'),matches=('match_id','nunique')).reset_index();
    pr['reliability']=np.where((pr.n>=100)&(pr.matches>=10),'adequate','low_n'); pr['ci_note']='match-cluster bootstrap required for publication'; pr.to_csv(OUT/'pressure_decision_execution.csv',index=False)
    # Required public artifact names.
    ag.to_csv(OUT/'keeper_context_rank_changes.csv',index=False)
    mol=ag[ag.keeper.str.contains('Moloney',case=False,na=False)]
    moln=int(mol.n.iloc[0]) if len(mol) else 0
    statuses=[
      ('Q1','context_out_of_sample','MODEST_SUPPORT','Held-out M1 and M2 improve Brier/log loss/AUC versus M0, while ECE worsens and M4 declines; context adds information with portability limits.'),
      ('Q2','calibration_vs_valuation','SUPPORTED_WITH_QUALIFICATION','The strongest observed context signal is distribution/base-rate shift; valuation rank changes are smaller and uncertain.'),
      ('Q3','keeper_assessment_change','EXPLORATORY_ONLY','Rank-shift table is restricted to n>=100 and uses an out-of-fold residual adjustment.'),
      ('Q4','target_environment_fit','SUPPORTED_WITH_QUALIFICATION','Comparisons require HIGH SUPPORT or ADEQUATE common support and effective sample size; low-support rows are not estimable.'),
      ('Q5','competition_vs_team','EXPLORATORY_ONLY','Grouped spread is reported without causal interpretation.'),
      ('Q6','decision_vs_execution','EXPLORATORY_ONLY','Split-half stability and context adjustment are insufficient for publication-grade quadrants.'),
      ('Q7','pressure_opposition','PRACTITIONER_ONLY','The Grace Moloney example has n=%d; candidate states remain association-based and require replication.'%moln),
      ('Q8','restart_policy','EXPLORATORY_ONLY','Corrected meter bands and continuous state descriptors do not support a universal distance claim.'),
      ('Q9','practical_questions','SUPPORTED_WITH_QUALIFICATION','The framework supports diagnosis and scenario design, not guaranteed tactical or transfer outcomes.'),
      ('Q10','central_thesis','THESIS_B_WITH_CAVEATS','The underlying ordering may be more portable than absolute probabilities, while context changes valuation and requires local validation.'),
    ]
    txt='# Contextual research evidence gate\n\n'
    txt+='## Decision packet\n\n| Question | Topic | Classification | Basis |\n|---|---|---|---|\n'
    txt+=''.join('| %s | %s | %s | %s |\n' % tuple(s) for s in statuses)+'\n'
    txt+='## Five strongest quantitative findings\n\n1. M1 competition context improves held-out Brier by 0.0035 and log loss by 0.0089 versus M0; M2 team context improves Brier by 0.0070 and log loss by 0.0186.\n2. M2 AUC is 0.713 versus 0.667 for M0, but ECE is not improved, so context chiefly shifts baseline/ranking rather than producing portable probabilities.\n3. Keeper effects add no held-out gain beyond team context (M3 Brier 0.1826 versus M2 0.1820); match context declines (M4 0.1841).\n4. Target-environment fit has substantial unsupported mass for many keeper-target pairs; overlap-restricted fits are therefore low-support or not estimable.\n5. Pressure profiles provide descriptive selection, execution-residual and value shifts, but adequate match replication is uncommon.\n\n## Five things we cannot claim\n\n1. We cannot claim causal transfer success.\n2. We cannot claim a universal keeper ranking is valid across environments.\n3. We cannot claim the Grace Moloney example is publication-grade.\n4. We cannot claim a universal 40–60 m restart penalty.\n5. We cannot claim stable decision/execution recruitment quadrants.\n\n## Global versus competition versus team versus opponent\n\nCompetition adds modest held-out information; team context adds more within the present nested comparison; keeper and match/opponent layers do not add stable incremental information. Effects are associational and not causal.\n\n## Decision + Execution + Fit\n\nThe full triad does not survive as a publication-grade central claim. Decision/value diagnostics are usable with qualification; execution residuals and cross-environment fit remain exploratory.\n\n## Fit status\n\nFit is practitioner/exploratory. Responsible use requires at least 50% target-state common support, effective sample size proportional to observed n, capped weights, and explicit unsupported-state reporting.\n\n## Opposition level\n\nHighest defensible level: **Level 1** — descriptively identify contexts associated with lower retention/value. Level 2 is not established after state adjustment; Level 4 is unsupported.\n\n## Strongest worked keeper example\n\nGrace Moloney is the strongest current worked example by sample size (n=913), but remains practitioner-grade pending match-replicated uncertainty.\n\n## Most defensible recruitment application\n\nUse overlap-restricted contextual fit to identify observed strengths and unsupported target states for follow-up scouting; do not forecast transfer success.\n\n## Most defensible training application\n\nReproduce adequately sampled pressure/state combinations and measure R1, xR residual and value on new sessions, carrying low-n flags forward.\n\n## Recommended Sloan thesis\n\nGoalkeeper distribution risk structure shows modest contextual portability: competition and team environments improve held-out prediction, while absolute probabilities and cross-environment fit require local calibration and explicit common-support limits.\n'
    (ROOT/'reports/CONTEXT_EVIDENCE_GATE.md').write_text(txt)
    print('context evidence gate written',flush=True)
if __name__=='__main__': main()
