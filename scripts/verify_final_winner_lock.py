from pathlib import Path
import json, pandas as pd, sys
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/final_winner_lock'; errors=[]
required=['deep_independent_final.csv','deep_transport_vs_refit_corrected.csv','deep_spatial_gradient_final.csv','criterion_validity_final.csv','ablation_final.csv','team_adjusted_final.csv','target_selection_final.csv','intended_geometry_final.csv','portability_final.csv','final_claim_ledger.csv','model_specification_table.csv','abstract_fact_set.json','final_figure_plan.csv']
for f in required:
 if not (R/f).exists(): errors.append('missing '+f)
# independent CIs must be genuine intervals, not copied points
if (R/'deep_independent_final.csv').exists():
 d=pd.read_csv(R/'deep_independent_final.csv')
 if (d['C/V_lower95']==d.C_over_V).all() or (d['C/V_upper95']==d.C_over_V).all(): errors.append('independent deep CI copied point estimates')
# corrected transport table must contain genuine intervals
if (R/'deep_transport_vs_refit_corrected.csv').exists():
 d=pd.read_csv(R/'deep_transport_vs_refit_corrected.csv')
 if (d.independent_lower95==d.independent_OOF_C_over_V).all() or (d.independent_upper95==d.independent_OOF_C_over_V).all(): errors.append('transport CI copied point estimates')
# spatial bootstrap metadata and bands
if (R/'deep_spatial_gradient_final.csv').exists():
 d=pd.read_csv(R/'deep_spatial_gradient_final.csv')
 if d.bootstrap_draws.nunique()!=1 or d.bootstrap_draws.iloc[0]!=10000: errors.append('spatial bootstrap draw count')
 if not d.multiplicity_retained.all(): errors.append('spatial match multiplicity not retained')
 if set(d.band)!= {'0-16.5m','16.5-35m','35-52.5m'}: errors.append('invalid spatial bands')
# claim safeguards
ledger=pd.read_csv(R/'final_claim_ledger.csv'); txt=' '.join(ledger.astype(str).values.ravel()).lower()
if 'full xt-gk is a superior downstream outcome predictor,pass' in txt: errors.append('full xT-GK called superior')
if 'action-specific c improves fixed/coarse failure penalties,pass' in txt: errors.append('action-specific C called superior')
if '2.19–2.28 transported magnitude,pass' in txt: errors.append('old magnitude marked pass')
facts=json.loads((R/'abstract_fact_set.json').read_text()); ftxt=' '.join(facts.values()).lower()
for banned in ['2.19–2.28','causal transfer success','full xt-gk is superior','action-specific c proven superior']:
 if banned in ftxt: errors.append('prohibited abstract phrase '+banned)
# target caveat required in intended geometry documentation
if 'outcome-dependently' not in (ROOT/'reports/CLUB_APPLICATION_WORKED_EXAMPLE.md').read_text(): errors.append('target selection caveat missing')

# independent audit consolidation checks
if (R/'criterion_validity_final.csv').exists():
 d=pd.read_csv(R/'criterion_validity_final.csv')
 ad=d[d.panel=='action-decile']
 if (ad['positive_cells']>20).any() or (ad['action_positive_cells']>20).any():
  errors.append('action-decile positive cell count exceeds 20 cohort×outcome cells')
if (R/'intended_geometry_final.csv').exists():
 d=pd.read_csv(R/'intended_geometry_final.csv')
 if not ((d['TA_GEO_AUC']-d['TF_AUC']-d['delta_AUC']).abs()<1e-9).all():
  errors.append('intended geometry delta_AUC is not TA-GEO minus TF')
 if (d['delta_AUC']<=0).any():
  errors.append('intended geometry AUC gain direction is not positive TA-GEO minus TF')
if (R/'target_selection_final.csv').exists():
 d=pd.read_csv(R/'target_selection_final.csv')
 if not ((d['target_identifiable_pct']>=0)&(d['target_identifiable_pct']<=100)).all():
  errors.append('invalid target-identifiable percentage')
 if not (d['target_identifiable_pct_R1_1'] > d['target_identifiable_pct_R1_0']).all():
  errors.append('target-selection retained/failure rates inconsistent with accepted outcome-selective source')

if errors:
 print('VERIFY FINAL WINNER LOCK: FAIL')
 print('\n'.join(errors)); sys.exit(1)
print('VERIFY FINAL WINNER LOCK: PASS')
