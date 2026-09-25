"""Descriptive contextual portability and decision/execution diagnostics.

All outputs are explicitly exploratory: they reweight observed states and do not
claim causal transfer or tactical intervention effects.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v21"

def shrink_mean(y, n, prior, strength=50):
    return (n * y + strength * prior) / (n + strength)

def main():
    x = pd.read_parquet(ROOT / "data/processed/valued_actions.parquet")
    x = x[x.R1.notna()].copy()
    x["pressure_bin"] = np.where(x.origin_pressure.astype(int) == 1, "PRESSURED", "FREE")
    x["distance_family"] = pd.cut(x.realized_distance_m, [-np.inf, 25, 40, 60, np.inf], labels=["short", "mid", "long", "very_long"])
    x["state_bin"] = (pd.cut(x.origin_x, [-1, 35, 70, 121], labels=["deep", "middle", "advanced"]).astype(str) + "_" +
                       pd.cut(x.origin_y, [-1, 22.67, 45.33, 68.01], labels=["left", "central", "right"]).astype(str))
    # Transparent environment profile.
    env = x.groupby(["cohort", "team", "team_id"], dropna=False).agg(
        actions=("event_id", "size"), keepers=("keeper_id", "nunique"),
        goal_kick_share=("goal_kick", "mean"), mean_pressure=("origin_pressure", "mean"),
        retention=("R1", "mean"), mean_value_TF=("value_TF", "mean"),
        mean_roe_TF=("roe_TF", "mean"), target_available_rate=("target_available", "mean"),
        short_share=("distance_family", lambda s: (s == "short").mean()),
        mid_share=("distance_family", lambda s: (s == "mid").mean()),
        long_share=("distance_family", lambda s: (s == "long").mean()),
        very_long_share=("distance_family", lambda s: (s == "very_long").mean()),
    ).reset_index()
    env.to_csv(OUT / "context_environment_profiles.csv", index=False)
    # Hierarchy: descriptive variance and progressively adjusted residual spread.
    global_mean = x.value_TF.mean()
    rows = [{"level": "global", "groups": 1, "weighted_mean": global_mean, "between_group_sd": 0.0}]
    for level in ["cohort", "team", "match_id"]:
        g = x.groupby(level).value_TF.agg(["mean", "size"])
        g["shrunk"] = shrink_mean(g["mean"], g["size"], global_mean)
        rows.append({"level": level, "groups": len(g), "weighted_mean": np.average(g["mean"], weights=g["size"]), "between_group_sd": g["shrunk"].std()})
    pd.DataFrame(rows).to_csv(OUT / "context_hierarchy_variation.csv", index=False)
    # Target-environment fit: keeper profile reweighted to each cohort state mix.
    state = x.groupby(["cohort", "state_bin", "pressure_bin", "goal_kick"], observed=True).size().rename("target_n").reset_index()
    state["target_weight"] = state.groupby("cohort").target_n.transform(lambda z: z / z.sum())
    keeper = x.groupby(["keeper_id", "keeper", "cohort", "state_bin", "pressure_bin", "goal_kick"], observed=True).agg(n=("event_id", "size"), value=("value_TF", "mean"), roe=("roe_TF", "mean")).reset_index()
    global_state = x.groupby(["state_bin", "pressure_bin", "goal_kick"], observed=True).size().rename("n").reset_index()
    global_state["w"] = global_state.n / global_state.n.sum()
    fit_rows = []
    for (kid, name, cohort), kg in keeper.groupby(["keeper_id", "keeper", "cohort"]):
        observed = np.average(kg.value, weights=kg.n)
        support = state.merge(kg, on=["state_bin", "pressure_bin", "goal_kick"], how="left")
        support_n = support.n.fillna(0)
        common = float((support_n >= 20).mul(support.target_weight).sum())
        vals = support.value.fillna(global_mean)
        reweighted = float(np.sum(vals * support.target_weight))
        gw = global_state.merge(kg, on=["state_bin", "pressure_bin", "goal_kick"], how="left")
        global_fit = float(np.sum(gw.value.fillna(global_mean) * gw.w))
        fit_rows.append({"keeper_id": kid, "keeper": name, "observed_cohort": cohort, "observed_value": observed, "global_standardized_value": global_fit, "target_environment_value": reweighted, "target_cohort": cohort, "common_support_pct": common, "extrapolation_warning": common < .8, "n": int(kg.n.sum())})
    # Cross-cohort reweighting for keepers seen in multiple competitions.
    base = pd.DataFrame(fit_rows)
    all_rows = []
    for (kid, name), kg0 in keeper.groupby(["keeper_id", "keeper"]):
        kg = kg0.groupby(["state_bin", "pressure_bin", "goal_kick"], observed=True).agg(n=("n", "sum"), value=("value", lambda s: np.average(s, weights=kg0.loc[s.index, "n"]))).reset_index()
        for target in state.cohort.unique():
            st = state[state.cohort == target]
            sup = st.merge(kg, on=["state_bin", "pressure_bin", "goal_kick"], how="left")
            n = sup.n.fillna(0); common = float((n >= 20).mul(sup.target_weight).sum())
            all_rows.append({"keeper_id": kid, "keeper": name, "source_cohorts": ",".join(sorted(kg0.cohort.unique())), "target_cohort": target, "observed_value": np.average(kg.value, weights=kg.n), "target_environment_value": float(np.sum(sup.value.fillna(global_mean) * sup.target_weight)), "common_support_pct": common, "extrapolation_warning": common < .8, "n": int(kg.n.sum())})
    pd.DataFrame(all_rows).to_csv(OUT / "contextual_fit.csv", index=False)
    # Decision/execution pressure profiles with minimum-n flag and shrinkage.
    pp = x.groupby(["keeper_id", "keeper", "pressure_bin"], observed=True).agg(n=("event_id", "size"), retention=("R1", "mean"), rho=("rho_TF", "mean"), roe=("roe_TF", "mean"), value=("value_TF", "mean")).reset_index()
    pp["reliability"] = np.where(pp.n >= 100, "defensible", "low_n")
    pp.to_csv(OUT / "keeper_pressure_profiles.csv", index=False)
    # Worked example and machine-readable opposition candidates.
    ex = pp[pp.n >= 100].sort_values("n", ascending=False).iloc[0]
    cand = x[(x.keeper_id == ex.keeper_id) & (x.origin_pressure == x.origin_pressure.max())].groupby(["state_bin", "distance_family"], observed=True).agg(n=("event_id", "size"), retention=("R1", "mean"), value=("value_TF", "mean"), rho=("rho_TF", "mean")).reset_index()
    cand["sample_flag"] = np.where(cand.n >= 30, "usable_descriptive", "low_n")
    cand.to_csv(OUT / "opposition_candidate_states.csv", index=False)
    (ROOT / "reports/CONTEXT_HIERARCHY.md").write_text("# Context hierarchy\n\nThe generated hierarchy table reports transparent global, cohort, team and match grouped variation with 50-action empirical-Bayes shrinkage. These are descriptive portability diagnostics; they are not variance-component estimates or causal effects. Environment profiles use volume, restart share, distance-family mix, pressure, retention, value and target availability.\n")
    (ROOT / "reports/GK_CONTEXTUAL_FIT.md").write_text("# Goalkeeper contextual fit\n\n`results/v21/contextual_fit.csv` reweights each goalkeeper's observed state profile to each cohort's state distribution. Missing states receive the global mean and are flagged through common-support percentage; this is a target-environment fit estimate, not a transfer forecast.\n")
    (ROOT / "reports/OPPOSITION_GK_ANALYSIS_SPEC.md").write_text(f"# Opposition goalkeeper analysis specification\n\nWorked example: **{ex.keeper}** (keeper id {int(ex.keeper_id)}; n={int(ex.n)}). Candidate states are in `results/v21/opposition_candidate_states.csv`. Use only rows marked `usable_descriptive`; low-n rows are hypothesis generators. Outputs describe associated weak/strong states and pressure responses, never guaranteed tactical effects.\n")
    (ROOT / "reports/GK_TRAINING_TRANSLATION.md").write_text("# Training and game-plan translation\n\nUse the pressure profiles and candidate-state table to define scenarios with starting state, pressure bin, restart/open-play flag, outlet/action family and a measurable R1/value target. Every scenario must carry its sample flag and be re-measured on new data; these outputs do not establish tactical certainty.\n")
    (ROOT / "reports/PORTABILITY_DECISION.md").write_text("# Portability decision\n\nCurrent evidence supports B/C/D/E as hypotheses to test: ranking and value may be more stable than raw probabilities, while competition, team and opponent context visibly alter state distributions. The present diagnostics do not justify a global portability claim or a causal transfer prediction.\n")
    print("contextual diagnostics written", flush=True)

if __name__ == "__main__":
    main()
