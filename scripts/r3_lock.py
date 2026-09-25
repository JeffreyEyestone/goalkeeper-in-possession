"""Generate the definitive R3 quantitative lock packet from canonical artifacts."""
from pathlib import Path
import sys
import json, shutil, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); OUT=ROOT/'results/v21_r3'; OUT.mkdir(exist_ok=True)
from gkpossession.xr import fit_calibration, calibrated
from gkpossession.stats import metrics
def main():
    x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet'); x=x[x.R1.notna()].copy(); x['rho']=x.rho_TF
    rows=[]
    for cohort,g in x.groupby('cohort'):
      for policy in ['strict_zero_shot','local_intercept','local_intercept_slope','local_isotonic']:
        if policy=='strict_zero_shot': p=g.rho.to_numpy()
        else:
          ids=np.array(sorted(g.match_id.unique())); halves=[ids[::2],ids[1::2]]; pred=np.full(len(g),np.nan)
          for h in range(2):
            calg=g[g.match_id.isin(halves[1-h])]; test=g[g.match_id.isin(halves[h])]; method='isotonic' if policy=='local_isotonic' else 'platt'; z=fit_calibration(calg.R1.astype(int).to_numpy(),calg.rho.to_numpy(),method); pred[g.index.get_indexer(test.index)]=calibrated(z,test.rho.to_numpy(),method)
          p=pred
        m=metrics(g.R1,p)
        rows.append({'cohort':cohort,'regime':policy,'scope':'held_out_match_grouped' if policy!='strict_zero_shot' else 'frozen_external','n':m.get('n'),'prevalence':m.get('base_rate'),'mean_prediction':m.get('mean_prediction'),'gap':m.get('mean_gap'),'auc':m.get('auc'),'pr_auc':m.get('pr_auc'),'brier':m.get('brier'),'log_loss':m.get('log_loss'),'ece':m.get('ece'),'calibration_intercept':m.get('calibration_intercept'),'calibration_slope':m.get('calibration_slope')})
      for k in [25,50,100,250,500]:
        rows.append({'cohort':cohort,'regime':f'learning_curve_{k}_actions','scope':'descriptive_calibration_learning_curve','n':min(k,len(g)),'prevalence':g.R1.head(k).mean(),'mean_prediction':g.rho.head(k).mean(),'gap':g.rho.head(k).mean()-g.R1.head(k).mean()})
      for k in [1,2,3,5]:
        mids=sorted(g.match_id.unique())[:k]; q=g[g.match_id.isin(mids)]; rows.append({'cohort':cohort,'regime':f'learning_curve_{k}_matches','scope':'descriptive_calibration_learning_curve','n':len(q),'prevalence':q.R1.mean(),'mean_prediction':q.rho.mean(),'gap':q.rho.mean()-q.R1.mean()})
    pd.DataFrame(rows).to_csv(OUT/'calibration_transport.csv',index=False)
    shutil.copy2(ROOT/'results/v21_r2/receiver_model_comparison.csv',OUT/'tf_ta_comparison.csv')
    d=pd.read_csv(OUT/'tf_ta_comparison.csv'); geo=d[d.model=='TA-GEO'].iloc[0]; role=d[d.model=='TA-ROLE'].iloc[0]; player=d[d.model=='TA-PLAYER'].iloc[0]
    pd.DataFrame([{'contrast':'ROLE_MINUS_GEO','delta_auc':role.auc-geo.auc,'delta_brier':role.brier-geo.brier,'delta_log_loss':role.log_loss-geo.log_loss,'delta_ece':role.ece-geo.ece},{'contrast':'PLAYER_MINUS_ROLE','delta_auc':player.auc-role.auc,'delta_brier':player.brier-role.brier,'delta_log_loss':player.log_loss-role.log_loss,'delta_ece':player.ece-role.ece},{'contrast':'PLAYER_MINUS_GEO','delta_auc':player.auc-geo.auc,'delta_brier':player.brier-geo.brier,'delta_log_loss':player.log_loss-geo.log_loss,'delta_ece':player.ece-geo.ece}]).to_csv(OUT/'receiver_incremental_changes.csv',index=False)
    for a,b in [('context_claim_tests.csv','context_claim_tests.csv'),('contextual_fit_support.csv','contextual_fit_support.csv'),('context_keeper_rank_shifts.csv','keeper_context_adjustment.csv'),('pressure_decision_execution.csv','pressure_decision_execution.csv')]: shutil.copy2(ROOT/'results/v21_r2'/a if (ROOT/'results/v21_r2'/a).exists() else ROOT/'results/v21'/a,OUT/b)
    pd.DataFrame([{'analysis':'corrected_standardized_restart','cohort':c,'status':'40_60_CLAIM_DISABLED','note':'retrospective realized distance; intended distance unavailable in TF'} for c in ['PL1516','T2024','WC2018','WC2022','WSL2021']]).to_csv(OUT/'restart_analysis.csv',index=False)
    reports={
      'SCIENTIFIC_STATUS_R3.md':'# Scientific status R3\n\nR3 locks the corrected calibration, cluster bootstrap, binary pressure and overlap-fit estimands. Context adds modest known-environment information; unseen-team/competition portability is unresolved. TA receiver role/identity improves selected-sample prediction modestly beyond geometry, but selection is outcome-dependent. Decision is supported with qualification; execution and fit remain practitioner/exploratory. No causal transfer, stable chemistry, universal restart curve or Level 4 opposition claim is supported.\n',
      'CALIBRATION_DECISION_R3.md':'# Calibration decision R3\n\nThe quantitative transport table reports strict zero-shot, local recalibration sensitivities and learning curves using the corrected logistic calibration helper. Modern TF zero-shot calibration remains failed; local recalibration is not a portability result.\n',
      'TARGET_FREE_VS_TARGET_AWARE_R3.md':'# TF versus TA R3\n\nTA is evaluated only on target-identifiable actions. Intended geometry is essential for the selected action-specific estimand; receiver role and identity add modest incremental information. TA cannot be generalized to all actions and its selection is outcome-dependent.\n',
      'RECEIVER_SELECTION_AUDIT_R3.md':'# Receiver selection audit R3\n\nReceiver identity is observed only when a unique UUID-linked Ball Receipt event is available. Availability varies by cohort, outcome and keeper; this selection mechanism is retained as a limitation and no receiver result is treated as all-action evidence.\n',
      'OPTION_SET_DECISION_R3.md':'# Option-set decision R3\n\nOpen event data identify the linked intended receiver but not all feasible alternatives at release. A defensible option-set model therefore requires tracking/360 locations, pressure, lanes and cover.\n'}
    for n,t in reports.items():(ROOT/'reports'/n).write_text(t)
    print('R3 lock packet written',flush=True)
if __name__=='__main__':main()
