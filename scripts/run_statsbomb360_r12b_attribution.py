"""Fixed nested OOF attribution test for frozen xR and StatsBomb 360.

All model feature sets are frozen in input_and_model_lock.json. Existing R1.2A
match folds are reused within ALL, WC2022 and EURO2024 scopes. No canonical
model is fitted or altered.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/statsbomb360_r12b"
SCOPES = ("ALL", "WC2022", "EURO2024")
MODELS = ("M0", "M1", "M2", "M3", "M4", "M2C", "M3C", "M2L", "M2E")
METRICS = ("AUC", "Brier", "log_loss")
N_BOOT = 5000
SEED = 20260924
BASE = ("logit_rho", "under_pressure", "goal_kick")
OPTIONS = ("lane_clear_short_options", "central_short_exit")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_input_lock() -> dict:
    lock = json.loads((OUT/"input_and_model_lock.json").read_text())
    for relative, expected in lock["input_sha256"].items():
        path = ROOT/relative
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return lock


def polygon_area_fraction(frame_json: str) -> tuple[float, float]:
    polygon = np.asarray(json.loads(frame_json).get("visible_area", []), dtype=float)
    if polygon.size < 6 or polygon.size % 2:
        return np.nan, np.nan
    p = polygon.reshape(-1, 2) * np.array([105/120, 68/80])
    area = .5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                      np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    return float(area), float(area/(105*68))


def weighted_auc_tie_aware(y: np.ndarray, score: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
    """Weighted Mann–Whitney AUC, with half credit to tied scores."""
    order = np.argsort(score, kind="stable")
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(score[order]) != 0)]
    pos = np.add.reduceat(weights[order]*y[order, None], boundaries, axis=0)
    neg = np.add.reduceat(weights[order]*(1-y[order, None]), boundaries, axis=0)
    prior_neg = np.cumsum(neg, axis=0)-neg
    return np.sum(pos*(prior_neg+.5*neg), axis=0)/(pos.sum(axis=0)*neg.sum(axis=0))


def spec_for_scope(scope: str) -> dict[str, tuple[str, ...]]:
    competition = ("comp_euro",) if scope == "ALL" else ()
    m1 = ("logit_rho",) + competition
    m2 = BASE + competition
    m3 = m2 + OPTIONS
    m2c = m2 + ("visible_area_fraction_of_pitch",)
    return {
        "M0": (), "M1": m1, "M2": m2, "M3": m3,
        "M4": m3 + ("receiver_option_count",),
        "M2C": m2c, "M3C": m2c + OPTIONS,
        "M2L": m2 + ("lane_clear_short_options",),
        "M2E": m2 + ("central_short_exit",),
    }


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_parquet(ROOT/"results/statsbomb360/distribution_option_summary.parquet")
    x = pd.read_parquet(ROOT/"results/winner_gate13/oof_action_values.parquet",
                        columns=["event_id", "rho_oof", "R1"])
    d = s.merge(x, on="event_id", how="inner", validate="one_to_one")
    if len(s) != 4831 or len(d) != 4802 or d.match_id.nunique() != 115:
        raise RuntimeError("Frozen R1.2 population changed")
    areas = [polygon_area_fraction(frame) for frame in d.frame_json]
    d["visible_area_m2"] = [a for a, _ in areas]
    d["visible_area_fraction_of_pitch"] = [f for _, f in areas]
    if not d.visible_area_fraction_of_pitch.between(0, 1).all():
        raise RuntimeError("Visible polygon area is absent or outside pitch bounds")
    d["rho_clip"] = d.rho_oof.clip(.001,.999)
    d["logit_rho"] = np.log(d.rho_clip/(1-d.rho_clip))
    d["comp_euro"] = (d.competition == "EURO2024").astype(int)
    fold = pd.read_parquet(ROOT/"results/statsbomb360_r12a/incremental_oof_predictions.parquet",
                           columns=["scope", "event_id", "match_id", "evaluation_fold"])
    assignments = []
    for scope in SCOPES:
        part = fold[fold.scope == scope]
        expected = set(d.event_id) if scope == "ALL" else set(d[d.competition == scope].event_id)
        if not part.event_id.is_unique or set(part.event_id) != expected:
            raise RuntimeError(f"R1.2A folds do not cover identical {scope} actions")
        if part.groupby("match_id").evaluation_fold.nunique().max() != 1:
            raise RuntimeError(f"Match crosses fold in {scope}")
        assignments.append(part)
    assignment = pd.concat(assignments, ignore_index=True)
    return d, assignment


def fit_scope(d: pd.DataFrame, fold: pd.DataFrame, scope: str) -> pd.DataFrame:
    part = d if scope == "ALL" else d[d.competition == scope]
    part = part.merge(fold[fold.scope == scope][["event_id", "evaluation_fold"]],
                      on="event_id", how="inner", validate="one_to_one").sort_values("event_id").reset_index(drop=True)
    if len(part) != (4802 if scope == "ALL" else len(d[d.competition == scope])):
        raise RuntimeError("Fold join lost actions")
    features = spec_for_scope(scope)
    if set(features["M3"])-set(features["M2"]) != set(OPTIONS):
        raise RuntimeError("Primary nested model is not exactly two 360 terms")
    predictions = {"M0": part.rho_oof.to_numpy(dtype=float)}
    y = part.R1.to_numpy(dtype=float)
    for name in MODELS[1:]:
        cols = list(features[name])
        pred = np.full(len(part), np.nan)
        for k in range(5):
            train = part.evaluation_fold != k
            test = part.evaluation_fold == k
            train_matches = set(part.loc[train, "match_id"])
            test_matches = set(part.loc[test, "match_id"])
            if train_matches.intersection(test_matches):
                raise RuntimeError("Match leakage")
            x_train = sm.add_constant(part.loc[train, cols].astype(float), has_constant="add")
            x_test = sm.add_constant(part.loc[test, cols].astype(float), has_constant="add")
            model = sm.GLM(y[train], x_train, family=sm.families.Binomial()).fit()
            pred[test] = model.predict(x_test)
        if not np.isfinite(pred).all() or not ((pred > 0) & (pred < 1)).all():
            raise RuntimeError(f"Invalid OOF prediction in {scope} {name}")
        predictions[name] = pred
    result = part[["event_id", "match_id", "competition", "R1", "rho_oof",
                   "rho_clip", "logit_rho", "visible_area_m2",
                   "visible_area_fraction_of_pitch", "evaluation_fold"]].copy()
    result.insert(0, "scope", scope)
    for name in MODELS:
        result[name] = predictions[name]
    if not np.array_equal(result.M0.to_numpy(), result.rho_oof.to_numpy()):
        raise RuntimeError("M0 was fitted or altered")
    return result


def metrics_for_scope(oof: pd.DataFrame, scope: str) -> pd.DataFrame:
    y = oof.R1.to_numpy(dtype=int)
    rows = []
    for name in MODELS:
        p = oof[name].to_numpy(dtype=float)
        values = (roc_auc_score(y,p), brier_score_loss(y,p), log_loss(y,p))
        for metric, value in zip(METRICS, values):
            rows.append({"scope": scope, "model": name, "metric": metric,
                         "value": value, "n": len(oof), "matches": oof.match_id.nunique(),
                         "feature_specification": "+".join(spec_for_scope(scope)[name]) or "frozen_rho_no_fit",
                         "fold_assignment": "same 5 match-grouped folds for every fitted model"})
    return pd.DataFrame(rows)


def paired_bootstrap(oof: pd.DataFrame, scope: str, smaller: str,
                     larger: str) -> pd.DataFrame:
    y = oof.R1.to_numpy(dtype=int)
    p = oof[smaller].to_numpy(dtype=float)
    q = oof[larger].to_numpy(dtype=float)
    matches = np.sort(oof.match_id.unique())
    match_code = pd.Categorical(oof.match_id, categories=matches).codes
    rng = np.random.default_rng(SEED + {"ALL": 0, "WC2022": 1, "EURO2024": 2}[scope])
    rows = []
    for start in range(0, N_BOOT, 100):
        sampled = rng.integers(0, len(matches), size=(100, len(matches)))
        counts = np.array([np.bincount(sample, minlength=len(matches)) for sample in sampled]).T
        w = counts[match_code]
        den = w.sum(axis=0)
        auc_p = weighted_auc_tie_aware(y,p,w)
        auc_q = weighted_auc_tie_aware(y,q,w)
        brier_p = np.sum(w*(y[:,None]-p[:,None])**2, axis=0)/den
        brier_q = np.sum(w*(y[:,None]-q[:,None])**2, axis=0)/den
        loss_p = -np.sum(w*(y[:,None]*np.log(p[:,None]) + (1-y[:,None])*np.log(1-p[:,None])), axis=0)/den
        loss_q = -np.sum(w*(y[:,None]*np.log(q[:,None]) + (1-y[:,None])*np.log(1-q[:,None])), axis=0)/den
        rows.extend({"scope": scope, "draw": start+i, "control": smaller, "augmented": larger,
                     "delta_auc": auc_q[i]-auc_p[i], "delta_brier": brier_q[i]-brier_p[i],
                     "delta_log_loss": loss_q[i]-loss_p[i], "matches_drawn": len(matches),
                     "distinct_matches": int((counts[:,i]>0).sum())} for i in range(100))
    return pd.DataFrame(rows)


def comparison_table(validation: pd.DataFrame, draws: pd.DataFrame,
                     smaller: str, larger: str) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        for metric, column in zip(METRICS, ("delta_auc", "delta_brier", "delta_log_loss")):
            a = validation[(validation.scope==scope)&(validation.model==smaller)&(validation.metric==metric)].iloc[0]
            b = validation[(validation.scope==scope)&(validation.model==larger)&(validation.metric==metric)].iloc[0]
            d = draws[draws.scope==scope]
            lo, hi = d[column].quantile([.025,.975])
            rows.append({"scope": scope, "control": smaller, "augmented": larger,
                         "metric": metric, "control_value": a.value, "augmented_value": b.value,
                         "delta": b.value-a.value, "ci_low": lo, "ci_high": hi,
                         "n": b.n, "matches": b.matches,
                         "bootstrap": "5000 paired match-cluster draws, multiplicity and tied AUC preserved"})
    return pd.DataFrame(rows)


def area_audit(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        part = d if scope == "ALL" else d[d.competition == scope]
        for variable in ("receiver_option_count", "lane_clear_short_options",
                         "central_short_exit", "R1"):
            rows.append({"scope": scope, "variable": variable, "n": len(part),
                         "area_mean_m2": part.visible_area_m2.mean(),
                         "area_fraction_mean": part.visible_area_fraction_of_pitch.mean(),
                         "area_fraction_min": part.visible_area_fraction_of_pitch.min(),
                         "area_fraction_max": part.visible_area_fraction_of_pitch.max(),
                         "pearson_r": part.visible_area_fraction_of_pitch.corr(part[variable]),
                         "spearman_r": part.visible_area_fraction_of_pitch.corr(part[variable],method="spearman")})
    return pd.DataFrame(rows)


def main() -> None:
    check_input_lock()
    d, fold = prepare_data()
    fold.sort_values(["scope","event_id"]).to_csv(OUT/"fold_assignments.csv", index=False)
    area_audit(d).to_csv(OUT/"visible_area_audit.csv", index=False)
    all_oof = []
    all_metrics = []
    primary_boot = []
    camera_boot = []
    for scope in SCOPES:
        result = fit_scope(d, fold, scope)
        all_oof.append(result)
        all_metrics.append(metrics_for_scope(result, scope))
        primary_boot.append(paired_bootstrap(result, scope, "M2", "M3"))
        camera_boot.append(paired_bootstrap(result, scope, "M2C", "M3C"))
        print(f"{scope}: {len(result)} identical OOF actions; M0–M4/M2C/M3C/ablations complete", flush=True)
    oof = pd.concat(all_oof, ignore_index=True)
    validation = pd.concat(all_metrics, ignore_index=True)
    primary = pd.concat(primary_boot, ignore_index=True)
    camera = pd.concat(camera_boot, ignore_index=True)
    oof.to_parquet(OUT/"nested_oof_predictions.parquet", index=False)
    validation.to_csv(OUT/"nested_model_validation.csv", index=False)
    primary.to_parquet(OUT/"primary_360_incremental_bootstrap.parquet", index=False)
    camera.to_parquet(OUT/"camera_control_bootstrap.parquet", index=False)
    comparison_table(validation, camera, "M2C", "M3C").to_csv(OUT/"camera_control_validation.csv", index=False)
    comparison_table(validation, primary, "M2", "M3").to_csv(OUT/"primary_attribution_validation.csv", index=False)
    baseline = validation[validation.model=="M2"][["scope","metric","value"]].rename(columns={"value":"m2_value"})
    ablation = validation[validation.model.isin(("M2L","M2E","M3"))].merge(baseline,on=["scope","metric"])
    ablation["delta_vs_m2"] = ablation.value-ablation.m2_value
    ablation["component"] = ablation.model.map({"M2L":"clear_short_only","M2E":"central_exit_only","M3":"both_core_360"})
    ablation.to_csv(OUT/"component_ablation.csv", index=False)
    print("R1.2B nested OOF attribution complete", flush=True)


if __name__ == "__main__":
    main()
