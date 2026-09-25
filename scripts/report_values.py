"""Generate gated value-stage diagnostics from ex-ante branch scores."""
from pathlib import Path
import json, sys, subprocess, datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def clustered_mean_ci(g, column, reps=500):
    groups = [z[column].to_numpy() for _, z in g.groupby("match_id")]
    rng = np.random.default_rng(42)
    sums = np.array([z.sum() for z in groups]); lens = np.array([len(z) for z in groups])
    draws = [sums[ix].sum()/lens[ix].sum() for ix in (rng.integers(0, len(groups), len(groups)) for _ in range(reps))]
    return float(g[column].mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main():
    out = ROOT / "results/v21"
    x = pd.read_parquet(ROOT / "data/processed/valued_actions.parquet")
    x = x[x.R1.notna()].copy()
    x["standardized_band"] = pd.cut(x.realized_distance_m, [-np.inf, 25, 40, 60, np.inf], right=False,
                                    labels=["<25", "25-40", "40-60", ">=60"])
    aggregates, relationships, bands, phi, kappa = [], [], [], [], []
    for cohort, g in x.groupby("cohort"):
        for est in ["TF", "TA"]:
            q = g if est == "TF" else g[g.target_available]
            for (kid, keeper), kg in q.groupby(["keeper_id", "keeper"]):
                if len(kg) < 25:
                    continue
                aggregates.append({"cohort": cohort, "estimand": est, "keeper_id": kid, "keeper": keeper,
                    "n": len(kg), "matches": kg.match_id.nunique(), "mean_rho": kg[f"rho_{est}"].mean(),
                    "raw_retention": kg.R1.mean(), "roe_raw": kg[f"roe_{est}"].mean(),
                    "value_per_action": kg[f"value_{est}"].mean(), "value_per_100": kg[f"value_{est}"].mean() * 100,
                    "mean_realized_distance_m": kg.realized_distance_m.mean(), "launch_share_native_gt60": (kg.native_distance > 60).mean(),
                    "goal_kick_share": kg.goal_kick.mean()})
        for est in ["TF", "TA"]:
            q = g if est == "TF" else g[g.target_available]
            k = pd.DataFrame([r for r in aggregates if r["cohort"] == cohort and r["estimand"] == est])
            if len(k) >= 3:
                relationships.extend([
                    {"cohort": cohort, "estimand": est, "relationship": "completion_vs_value", "method": "spearman", "estimate": spearmanr(k.raw_retention, k.value_per_action).statistic, "n": len(k)},
                    {"cohort": cohort, "estimand": est, "relationship": "realized_distance_vs_value", "method": "pearson", "estimate": pearsonr(k.mean_realized_distance_m, k.value_per_action).statistic, "n": len(k)},
                ])
            for band, kg in q[q.goal_kick == 1].groupby("standardized_band", observed=True):
                if len(kg):
                    mean, lo, hi = clustered_mean_ci(kg, f"value_{est}")
                    bands.append({"cohort": cohort, "estimand": est, "units": "standardized_pitch_m", "band": str(band), "n": len(kg),
                        "keepers": kg.keeper_id.nunique(), "value_mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "rho_mean": kg[f"rho_{est}"].mean(), "retention": kg.R1.mean(), "target_available_rate": kg.target_available.mean()})
        for band, kg in g.groupby(pd.cut(g.origin_x, [-1, 30, 60, 90, 121], labels=["deep", "middle", "advanced", "attacking"]), observed=True):
            if len(kg):
                phi.append({"cohort": cohort, "band": str(band), "n": len(kg), "mean_origin_value": kg.origin_value_TF.mean(),
                    "mean_failure_cost": kg.failure_cost_TF.mean(), "phi_observed_branch_ratio": kg.failure_cost_TF.mean() / max(kg.origin_value_TF.mean(), 1e-9)})
        q = g.groupby(["keeper_id", "keeper"]).agg(base=("value_TF", "mean"), **{f"k{k}": (f"value_TF_k{k}", "mean") for k in ["1.25", "1.5", "1.75", "2"]}).reset_index()
        for col in ["k1.25", "k1.5", "k1.75", "k2"]:
            if len(q) >= 3:
                kappa.append({"cohort": cohort, "comparison": col, "keepers": len(q), "spearman_vs_k1": spearmanr(q.base, q[col]).statistic})
    for name, data in [("keeper_aggregates", aggregates), ("value_relationships", relationships), ("restart_value_bands", bands), ("phi_branch_ratios", phi), ("kappa_sensitivity", kappa)]:
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    headline = [
        ("H1", "Completion does not measure distribution value", "QUALIFIED_PENDING_CONTEXT", "Completion/value associations are cohort-dependent; full contextual uncertainty suite remains pending."),
        ("H2", "Deep turnover consequence is asymmetric", "PENDING", "Frozen branch ratios are diagnostic; independent phi* inference with uncertainty remains pending."),
        ("H3", "Retention model transfers", "FAIL", "Strict TF modern transfer gaps are -0.105 to -0.117; ECE .118-.126."),
        ("H4", "Retention Over Expected contains repeatable signal", "QUALIFIED", "Preliminary league TF residual reliability is low/moderate and needs uncertainty/context."),
        ("H5", "Decision value is repeatable", "PENDING", "Aggregates generated; final clustered league reliability suite remains pending."),
        ("H6", "40-60 m restart penalty", "FAIL_TO_VERIFY", "Corrected standardized-meter analysis disables the historical meter-labeled claim pending contextual inference."),
        ("H7", "Style neutrality", "PENDING", "Realized-distance associations are descriptive; context adjustment remains pending."),
    ]
    pd.DataFrame([{"claim_id": h, "claim": c, "status": s, "evidence": e, "specification_id": "scientific-v2.1-TF-TA-amendment-20260916", "git_commit": commit, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()} for h, c, s, e in headline]).to_csv(out / "headline_claim_tests.csv", index=False)
    manifest = {
        "status": "exploratory_value_stage_checkpoint",
        "estimands": ["TF", "TA"],
        "label": "R1",
        "source_commit": commit,
        "calibration_gate": "FAILED_EXTERNAL_TF_CALIBRATION",
        "artifacts": [
            "branch_model_metrics.csv", "branch_model_selection.json", "keeper_aggregates.csv",
            "value_relationships.csv", "restart_value_bands.csv", "phi_branch_ratios.csv",
            "kappa_sensitivity.csv", "headline_claim_tests.csv"
        ],
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out / "value_results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "reports/VALIDATION_REPORT.md").write_text("# Validation report - gated value-stage checkpoint\n\nValue scores, branch diagnostics, standardized restart bands and keeper aggregates are generated, but strict TF calibration failure means they are exploratory. H3 fails under frozen transfer; H6 is disabled pending corrected contextual analysis. H1, H2, H4, H5 and H7 remain qualified or pending. Full clustered value uncertainty, baselines, context adjustment, phi* replication and final manuscript claims remain incomplete.\n")
    (ROOT / "reports/ROBUSTNESS_MATRIX.md").write_text("# Robustness matrix - current checkpoint\n\n| Axis | Status |\n|---|---|\n| TF/TA, R1/R2/R3/R15, classifiers, isotonic/Platt | Implemented and reported\n| Ex-ante branch payoffs and failure costs | Implemented; uncertainty pending\n| 16x12 surface | Implemented; grid sensitivity pending\n| kappa 1-2 | Generated; inference pending\n| Standardized meters/native sensitivity | Implemented; continuous/context model pending\n| Baselines and headline gates | Partial; final comparison pending\n| External TF calibration | Failed; no-refit claim disabled\n| Final paper results | Not generated\n")
    (ROOT / "reports/FINAL_SCIENTIFIC_SUMMARY.md").write_text("# Final scientific summary - interim\n\nTarget-free release-context/action-family information supports modest all-action discrimination, but frozen probabilities do not transfer as calibrated weights to modern cohorts. Intended-target information improves prediction on the identifiable subset, with outcome-dependent selection. Ex-ante branch scoring is implemented and ignores realized endpoints/outcomes, but value conclusions remain exploratory under the calibration failure. The historical meter-labeled 40-60 m claim is disabled.\n\nRemaining gates are calibration scope, final TF/TA value structure, clustered uncertainty, context-adjusted restart analysis, robustness and manuscript audit completion.\n")


if __name__ == "__main__":
    main()
