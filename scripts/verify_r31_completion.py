from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/r3'
reports=['SCIENCE_LOCK_R31.md','CALIBRATION_TRANSPORT_R31.md','PORTABILITY_R31.md','CONTEXT_FIT_R31.md','RECEIVER_MODEL_R31.md','GK_RECEIVER_CREDIT_R31.md','BUMP_PROXY_R31.md','LONG_TARGET_R31.md','PRESSURE_OPPOSITION_R31.md','RESTART_R31.md','MANUSCRIPT_CLAIM_AUDIT_R31.md','REPRODUCIBILITY_R31.md']
results=['calibration_transport.csv','calibration_learning_curves.csv','context_known_environment.csv','context_unseen_team.csv','context_unseen_competition.csv','contextual_fit.csv','keeper_context_adjustment.csv','pressure_effects.csv','receiver_model_comparison.csv','receiver_incremental_ci.csv','receiver_profile_coverage.csv','gk_receiver_components.csv','gk_receiver_pair_effects.csv','bump_proxy.csv','long_target_proxy.csv','restart_analysis.csv','headline_claim_tests.csv','paper_results_candidate.json']
def main():
 miss=[p for p in reports if not (ROOT/'reports'/p).exists()]+[p for p in results if not (R/p).exists()]
 assert not miss,miss
 d=pd.read_csv(R/'contextual_fit.csv'); assert d.target_cohort.nunique()==5; assert d.common_support_pct.between(0,1).all(); assert (d.unsupported_state_mass.between(0,1)).all()
 p=pd.read_csv(R/'pressure_effects.csv'); assert {'decision_effect_ci_lo','decision_effect_ci_hi','execution_effect_ci_lo','value_effect_ci_lo'}<=set(p.columns)
 assert 'M4_match_context' not in (ROOT/'scripts/context_evidence_gate.py').read_text()
 forbidden='isin'+'(draw)'; assert forbidden not in '\n'.join(p.read_text() for p in ROOT.glob('scripts/*.py') if p.name!='verify_r31_completion.py')
 assert 'copy2' not in (ROOT/'scripts/r31_finalize.py').read_text()
 print('R3.1 verification PASS')
if __name__=='__main__':main()
