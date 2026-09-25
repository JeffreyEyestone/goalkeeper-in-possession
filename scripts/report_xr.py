import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/gk_mplconfig')
from pathlib import Path
import json,sys,hashlib,subprocess,datetime
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/v21';figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
x=pd.read_parquet(ROOT/'data/processed/xr_predictions.parquet');m=pd.read_csv(out/'xr_metrics.csv');selected=json.loads((out/'xr_selected_models.json').read_text());sm=m[m.apply(lambda r:r.family==selected[r.estimand]['family'],axis=1)];bounds=pd.read_csv(out/'probability_boundaries.csv')
lines=['# xR model card — TF and TA, calibration review checkpoint','','Status: three classifier families and two calibrators have been evaluated. TF and TA remain separate. The primary isotonic TF variant triggers predeclared external calibration review in all four external cohorts. Three modern cohorts also fail the mean-gap/ECE thresholds. This is not a failure of TF discrimination and does not establish that open event data cannot support policy valuation. No xT or failure-cost model has yet been fit.', '',
'## Design and sample discipline','',
'R1 predicts own-team first controlled receipt/action or explicitly annotated restart rights at distribution resolution. See RETENTION_LABEL_DECISION.md for exact rules, unresolved cases and R1/R2/R3 sensitivities. TF uses origin, annotated origin pressure, goal-kick status, period/time, prior score state and body part. TA additionally uses UUID-linked intended receipt coordinates/distance/angle; no endpoint imputation or target-missingness predictor. Height, technique and play-pattern fields are withheld under strict release-time timing rules.', '',
'Frozen WC2018 split: 38 training / 13 calibration / 13 test matches, seed 42. No match crosses partitions. The model family is selected by four-fold match-grouped training Brier, before looking at held-out/external results. Logistic regression wins both TF and TA; the shallow tree and historical HistGradientBoosting configuration remain reported benchmarks. No class weighting. Isotonic and Platt each fit only the dedicated calibration partition. No external refitting occurs. Training/calibration rows are excluded from reported WC2018 evaluation.', '',
'All scalar metrics are in xr_metrics.csv, including n, base rate, prediction mean, signed gap, SD, ROC AUC, PR AUC (average precision), Brier, log loss, ECE, calibration intercept and slope. Calibration/slope and other selected-model intervals use 500 match-cluster bootstrap replicates with seed 42. ECE uses ten fixed equal-width bins including p=1 in the last bin. Reliability-bin intervals are Wilson descriptive intervals, not clustered bands. Log-loss reporting clips probabilities at 1e-6; slope/intercept logit inputs use 1e-5. Those numerical conventions matter for isotonic boundaries.', '',
'## Selected-model performance','', '| Estimand | Calibrator | Cohort | n | AUC | Brier | Clipped log loss | ECE | Mean gap | Slope |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in sm.itertuples():lines.append(f'| {r.estimand} | {r.calibration} | {r.cohort} | {r.n} | {r.auc:.3f} | {r.brier:.4f} | {r.log_loss:.4f} | {r.ece:.4f} | {r.mean_gap:+.4f} | {r.calibration_slope:.3f} |')
lines+=['','## Boundary probabilities and the scientific gate','','Inspect the selected-model CSV for information-gate diagnostics. Calibration gates use the pre-fit bounds in config/analysis.json. Slopes are sensitive to clipped logit inputs when isotonic returns zero or one; calibration_slope_clip_sensitivity.csv exposes this dependence. ECE, signed mean gap, Brier and boundary outcomes should be considered jointly rather than treating a slope flag as a convention-independent collapse.', '', '| Cohort | Predicted zero | Positive at zero | Predicted one | Negative at one | Raw log loss |','|---|---:|---:|---:|---:|---|']
for r in bounds.itertuples():lines.append(f'| {r.cohort} | {r.p_zero_n} | {r.positive_at_zero} | {r.p_one_n} | {r.negative_at_one} | {r.raw_log_loss} |')
lines+=['','The un-clipped external isotonic log loss is infinite wherever a certain prediction is contradicted. Finite CSV log losses are explicitly clipped computational summaries. In the value formula, rho=1 would entirely suppress expected failure cost, even though failures occur in that probability stratum. This is why the calibration issue requires review before valuing distributions.', '',
'## Predeclared Platt sensitivity','',
'The tables report the predeclared Platt sensitivity without any external refitting. Any promotion to primary after inspecting performance is a data-informed amendment; preserve the original isotonic assessment and avoid treating observed external cohorts as fresh confirmatory evidence. The strict release-time fit excludes height after a development run identified that its definition refers to realized peak flight. That run is archived under results/development/height_proxy_attempt; the revised evaluation discloses reuse of the same test/external cohorts.', '',
'## Limits','',
'No expert/video inter-rater labeling study has been performed. Ten real restart sequences are retained as inspected regression fixtures, supplemented by synthetic edge cases. Collection versions differ across eras. TF predictions for unresolved labels are extrapolations, and keeper/team attribution is inseparable here. TA benefits from intended location but samples a selected population with outcome-dependent target availability. No keeper execution-skill or final valuation claim follows from this model card.']
(ROOT/'reports/XR_MODEL_CARD.md').write_text('\n'.join(lines)+'\n')
common=pd.read_csv(out/'tf_ta_overlap_metrics.csv');rank=pd.read_csv(out/'tf_ta_probability_rank_comparison.csv');rep=pd.read_csv(out/'preliminary_xr_repeatability.csv')
lines=['# TF versus TA — probability-stage comparison','','This report is partial because the primary calibration review precedes value-surface/failure-cost/valuation fitting. Both models below are evaluated on the **same target-identifiable, R1-resolved sample**, with WC2018 training/calibration matches excluded. All-action TF performance is separately reported in XR_MODEL_CARD.md. No target imputation is performed.', '', '| Cohort | Estimand | Calibrator | Common n | AUC | Brier | ECE | Gap |','|---|---|---|---:|---:|---:|---:|---:|']
for r in common.itertuples():lines.append(f'| {r.cohort} | {r.estimand} | {r.calibration} | {r.n} | {r.auc:.3f} | {r.brier:.4f} | {r.ece:.4f} | {r.mean_gap:+.4f} |')
lines+=['','## Keeper probability/residual structure','','Cohort-specific keeper aggregates require at least 25 common-sample actions in tournaments, 200 in leagues, and two matches. Values below compare primary isotonic probability means and raw retention residual means, not xT values. They are exploratory point estimates, without bootstrap rank intervals at this checkpoint.', '', '| Cohort | Metric | Keepers | TF–TA Spearman |','|---|---|---:|---:|']
for r in rank.itertuples():lines.append(f'| {r.cohort} | {r.metric} | {r.keepers} | {r.spearman:.3f} |')
lines+=['','The strict models can also give different residual rankings: the common-sample table must not be interpreted as interchangeability of TF and TA. TF residuals still contain unmodeled destination/action-selection differences. These probability-stage differences are an early structural warning; xT value structure itself has not been calculated.', '', '## Preliminary league repeatability','','Frozen match-level random halves, seed 42; keepers need at least 100 resolved actions per half. Full and common-sample results are in preliminary_xr_repeatability.csv. These exploratory correlations lack uncertainty and are not final H4 tests or evidence of pure keeper skill. Any weak or negative residual result is preserved.', '', '| Cohort | Sample | Estimand | Measure | Keepers | Pearson | Spearman–Brown |','|---|---|---|---|---:|---:|---:|']
for r in rep.itertuples():lines.append(f'| {r.cohort} | {r.sample} | {r.estimand} | {r.metric} | {r.keepers} | {r.pearson:.3f} | {r.spearman_brown:.3f} |')
lines+=['','Target information can improve conditional probability prediction on the observed subset; this does not demonstrate what it would add on the disproportionately failed actions with missing targets. TF/TA value aggregates, value-rank correlations, H1–H7 relationships and corrected restart/repeatability comparisons remain pending. Gate B concerns value structure and cannot yet be evaluated. No disagreement is suppressed by treating an unrun value analysis as agreement.']
(ROOT/'reports/TARGET_FREE_VS_TARGET_AWARE.md').write_text('\n'.join(lines)+'\n')
# Figure-data table: generated from model outputs, never hand-entered values.
reliability=[]
for c,g in x[x.R1.notna()&~x.partition.isin(['train','calibration'])].groupby('cohort'):
    for method in ['isotonic','platt']:
        p=g[f'TF_{selected["TF"]["family"]}_{method}'].to_numpy();y=g.R1.to_numpy();bins=np.minimum((p*10).astype(int),9)
        for b in range(10):
            mask=bins==b;n=int(mask.sum())
            if n:
                q=y[mask].mean();den=1+1.96**2/n;center=(q+1.96**2/(2*n))/den;rad=1.96*np.sqrt(q*(1-q)/n+1.96**2/(4*n*n))/den
                reliability.append({'cohort':c,'calibration':method,'bin':b,'n':n,'predicted':p[mask].mean(),'observed':q,'lo':center-rad,'hi':center+rad})
r=pd.DataFrame(reliability);r.to_csv(out/'calibration_figure_data.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for ax,c in zip(axes.flat,['WC2018','WC2022','T2024','PL1516','WSL2021']):
    ax.plot([0,1],[0,1],color='.6',ls='--',lw=1)
    for method,color in [('isotonic','#2166ac'),('platt','#b35806')]:
        g=r[(r.cohort==c)&(r.calibration==method)]
        ax.errorbar(g.predicted,g.observed,yerr=[g.observed-g.lo,g.hi-g.observed],fmt='o-',markersize=3,color=color,lw=1,label=method.capitalize(),capsize=2)
    n=int(sm[(sm.estimand=='TF')&(sm.calibration=='isotonic')&(sm.cohort==c)].n.iloc[0]);ax.set(title=f'{c}   n={n:,}',xlabel='Predicted retention probability',ylabel='Observed retention',xlim=(-.03,1.03),ylim=(-.03,1.03))
axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='upper left',frameon=False)
axes.flat[-1].text(0,.65,'TF • frozen WC2018 fit\nWC2018: held-out matches only\nOther cohorts: no refitting\n\nTen fixed probability bins\n95% Wilson bin intervals\nR1 includes controlled receipt\nand annotated restart rights\n\nCalibration review checkpoint',va='top',transform=axes.flat[-1].transAxes)
fig.suptitle('Target-free retention calibration',fontsize=15)
for ext in ['png','pdf','svg']:fig.savefig(figdir/f'xr_tf_calibration_review.{ext}',dpi=200)
plt.close(fig)
(figdir/'xr_tf_calibration_review_caption.md').write_text('TF logistic model, trained on 38 WC2018 matches and calibrated on 13 separate matches. WC2018 test contains 13 unseen matches; external cohorts use frozen mappings. Ten equal-width bins include predictions of one. Error bars are 95% Wilson intervals (not clustered bootstrap bands). R1 means own controlled resolution or explicitly awarded restart rights. Boundary predictions and their contradictory outcomes are tabulated in results/v21/probability_boundaries.csv. This diagnostic is not a final paper figure.\n')
# Canonical probability-stage manifest: explicitly partial, with per-number provenance.
source=json.loads((ROOT/'config/data_source.json').read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();generated=datetime.datetime.now(datetime.timezone.utc).isoformat();numbers=[]
for row in m.to_dict('records'):
    for metric,value in row.items():
        if metric in ['estimand','family','calibration','cohort','sample','n'] or metric.endswith(('_lo','_hi')) or not isinstance(value,(float,int)) or not np.isfinite(value):continue
        lo=row.get(metric+'_lo');hi=row.get(metric+'_hi')
        numbers.append({'metric':metric,'cohort':row['cohort'],'estimand':row['estimand'],'family':row['family'],'calibration':row['calibration'],'sample':row['sample'],'n':row['n'],'estimate':value,'interval':[lo,hi] if lo is not None and np.isfinite(lo) and np.isfinite(hi) else None,'specification_id':'scientific-v2.1-TF-TA-amendment-20260916','data_manifest_hash':source['manifests']['matches_'+row['cohort']+'.json'],'git_commit':commit,'generated_at':generated,'code_function':'scripts/fit_xr.py:main; scripts/analyze_xr.py; src/gkpossession/stats.py:metrics'})
manifest={'status':'partial_xr_calibration_review_not_final_paper_results','numbers':numbers,'source_revision':source['revision'],'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'model_fit_code_commit':'fb148c4','source_code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list((ROOT/'src/gkpossession').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'config').glob('*.json')))},'output_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}}
(out/'xr_results_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
print('Wrote model cards, common-sample comparison, figure and partial canonical results manifest.')
