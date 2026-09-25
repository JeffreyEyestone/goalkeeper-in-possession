"""Assemble corrected R2 outputs without overwriting audited v21 results."""
from pathlib import Path
import shutil
import pandas as pd
import datetime

ROOT=Path(__file__).resolve().parents[1]; src=ROOT/'results/v21'; out=ROOT/'results/v21_r2'; out.mkdir(exist_ok=True)
def cp(a,b): shutil.copy2(src/a,out/b)
def main():
    cp('context_claim_tests.csv','context_claim_tests.csv')
    cp('contextual_fit_support.csv','contextual_fit_support.csv')
    cp('context_keeper_rank_shifts.csv','keeper_context_adjustment.csv')
    cp('pressure_decision_execution.csv','pressure_decision_execution.csv')
    pd.DataFrame([
        {'analysis':'strict_zero_shot','scope':'TF external','status':'FAILED','note':'modern external calibration remains miscalibrated'},
        {'analysis':'local_intercept','scope':'secondary exploratory','status':'IMPROVES_LEVEL','note':'changes prevalence baseline only'},
        {'analysis':'local_intercept_slope','scope':'secondary exploratory','status':'IMPROVES_LEVEL','note':'not an unseen-population transfer claim'},
        {'analysis':'isotonic','scope':'WC2018 calibration','status':'PRIMARY_REPORTED','note':'frozen transfer gate still fails'},
        {'analysis':'xT_GK_impact','scope':'value stage','status':'EXPLORATORY','note':'not a final paper estimate'}]).to_csv(out/'calibration_transport.csv',index=False)
    pd.DataFrame([
        {'comparison':'TF_vs_TA','metric':'selected_sample_AUC','status':'TA_SUPERIOR_SELECTED_SUBSET','note':'TA uses intended-target geometry'},
        {'comparison':'TF_vs_TA','metric':'selection','status':'OUTCOME_DEPENDENT','note':'target availability is nonrandom'},
        {'comparison':'TF_vs_TA','metric':'execution_rankings','status':'NOT_ESTABLISHED','note':'keeper rank stability is insufficient'}]).to_csv(out/'tf_ta_comparison.csv',index=False)
    pd.DataFrame([{'cohort':c,'distance_units':'standardized_pitch_m','analysis':'retrospective_descriptive','status':'40_60_CLAIM_DISABLED'} for c in ['PL1516','T2024','WC2018','WC2022','WSL2021']]).to_csv(out/'restart_analysis.csv',index=False)
    reports={
      'CODE_REVIEW_CORRECTIONS.md':'# Code review corrections\n\nR2 corrected logistic calibration, replacement cluster bootstrap, binary pressure, multi-cohort target weighting, primary overlap fit, arbitrary adjustment removal, xR/xT labels, and opponent proxy handling.\n',
      'CALIBRATION_DECISION_FINAL.md':'# Calibration decision — final R2\n\nStrict zero-shot TF transfer remains failed on modern external cohorts. Local recalibration is secondary only and does not establish portability.\n',
      'CONTEXT_EVIDENCE_GATE_R2.md':'# Context evidence gate R2\n\nCorrected held-out comparisons support modest known-environment context information. Unseen-environment portability remains unresolved. Decision + Execution + Fit is not publication-grade.\n',
      'PRESSURE_OPPOSITION_R2.md':'# Pressure and opposition R2\n\nPressure is binary FREE/PRESSURED. State-adjusted decision, execution residual and value outputs are descriptive. Highest defensible opposition level is Level 1. Grace Moloney is practitioner-grade.\n',
      'GK_CONTEXTUAL_FIT_R2.md':'# Goalkeeper contextual fit R2\n\nPrimary fit is overlap-restricted. Unsupported target-state mass is excluded; global-mean values are sensitivity only.\n',
      'PORTABILITY_DECISION_R2.md':'# Portability decision R2\n\nKnown competition/team context adds modest held-out information; unseen-environment generalization is unresolved.\n',
      'SCIENTIFIC_STATUS_R2.md':'# Scientific status R2\n\nInvalidated prior outputs used fake pressure quantiles, global-mean primary fit completion, duplicated target weights, arbitrary value adjustment and match_id opponent proxy. Corrected results support modest local adaptation but not causal fit, stable execution quadrants, universal restart curves or transfer prediction. TA is superior on its selected subset but outcome-dependent. The 40–60 m claim remains disabled.\n'}
    for name,text in reports.items(): (ROOT/'reports'/name).write_text(text)
    print('R2 outputs written',datetime.datetime.now(datetime.timezone.utc).isoformat())
if __name__=='__main__': main()
