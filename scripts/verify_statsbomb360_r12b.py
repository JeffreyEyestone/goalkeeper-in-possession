"""Hard R1.2B verifier. Default mode checks fresh raw JSON; --archive checks ZIP evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from run_statsbomb360_r12b_attribution import (
    MODELS, N_BOOT, OPTIONS, OUT, ROOT, SCOPES, spec_for_scope, weighted_auc_tie_aware,
)

RAW = Path('/private/tmp/statsbomb360_r12b_fresh_20260924')


def require(test, message: str) -> None:
    if not test:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def gate_direction(frame: pd.DataFrame, scope: str) -> bool:
    x=frame[frame.scope==scope]
    return len(x)==3 and all((r.delta>0 if r.metric=='AUC' else r.delta<0) for r in x.itertuples())


def main(archive_mode: bool=False) -> None:
    required = [
        'input_and_model_lock.json','fold_assignments.csv','nested_model_validation.csv',
        'nested_oof_predictions.parquet','primary_360_incremental_bootstrap.parquet',
        'primary_attribution_validation.csv','camera_control_validation.csv',
        'camera_control_bootstrap.parquet','component_ablation.csv','visible_area_audit.csv',
        'source_manifest_verified.csv','raw_to_derived_reproduction.csv','promotion_decision.json',
        'provenance.json',
    ]
    for name in required:
        require((OUT/name).is_file(),f'Missing R1.2B file: {name}')
    report=ROOT/'reports/STATSBOMB360_R12B_FINAL_ATTRIBUTION.md'
    require(report.is_file(),'Missing final attribution report')
    lock=json.loads((OUT/'input_and_model_lock.json').read_text())
    for relative,expected in lock['input_sha256'].items():
        require(sha256(ROOT/relative)==expected,f'Frozen input changed: {relative}')
    require(N_BOOT==5000,'Bootstrap draw count changed')
    for scope in SCOPES:
        spec=spec_for_scope(scope)
        require(spec['M3']==spec['M2']+OPTIONS,'M2/M3 differ by other than the two 360 terms')
        require(spec['M3C']==spec['M2C']+OPTIONS,'Camera-control models not nested')
        require(('comp_euro' in spec['M2'])==(scope=='ALL'),'Competition term misused')
        require(spec['M1']==(('logit_rho','comp_euro') if scope=='ALL' else ('logit_rho',)),
                'Local calibration specification changed')
    # Explicit AUC tie and repeated-row equivalence.
    y=np.array([0,1,1,0,0,1]); score=np.array([.1,.3,.3,.3,.8,.8]);w=np.array([2,1,3,4,1,2])
    require(np.isclose(weighted_auc_tie_aware(y,score,w[:,None])[0],
                       roc_auc_score(np.repeat(y,w),np.repeat(score,w))),
            'Tie-aware AUC or match multiplicity is wrong')

    folds=pd.read_csv(OUT/'fold_assignments.csv')
    oof=pd.read_parquet(OUT/'nested_oof_predictions.parquet')
    metrics=pd.read_csv(OUT/'nested_model_validation.csv')
    require(set(oof.scope)==set(SCOPES) and set(oof.columns).issuperset(set(MODELS)),
            'Missing nested predictions')
    for scope in SCOPES:
        d=oof[oof.scope==scope]
        f=folds[folds.scope==scope]
        require(d.event_id.is_unique and set(d.event_id)==set(f.event_id),
                f'Model action IDs differ in {scope}')
        require(d.match_id.nunique()==(115 if scope=='ALL' else 64 if scope=='WC2022' else 51),
                f'Match count changed in {scope}')
        merged=d.merge(f[['event_id','evaluation_fold']],on='event_id',suffixes=('_prediction','_assignment'))
        require((merged.evaluation_fold_prediction==merged.evaluation_fold_assignment).all(),
                f'Folds differ across nested models in {scope}')
        require(d.evaluation_fold.nunique()==5 and d.groupby('match_id').evaluation_fold.nunique().max()==1,
                f'Match fold leakage in {scope}')
        require(np.array_equal(d.M0.to_numpy(),d.rho_oof.to_numpy()),'M0 was modified')
        require(np.allclose(d.logit_rho,np.log(d.rho_clip/(1-d.rho_clip))),
                'Logit-rho transform absent')
        require(d[list(MODELS)].notna().all().all(),'A model lacks OOF predictions')
        for model in MODELS:
            p=d[model].to_numpy(); outcome=d.R1.to_numpy()
            expected={'AUC':roc_auc_score(outcome,p),'Brier':brier_score_loss(outcome,p),
                      'log_loss':log_loss(outcome,p)}
            for metric,val in expected.items():
                reported=metrics[(metrics.scope==scope)&(metrics.model==model)&(metrics.metric==metric)]
                require(len(reported)==1 and np.isclose(reported.value.iloc[0],val,atol=1e-12),
                        f'OOF {scope} {model} {metric} does not match predictions')
    require(len(oof[oof.scope=='ALL'])==4802,'Exact frozen-xR join changed')
    for name,control,augmented in (
        ('primary_360_incremental_bootstrap.parquet','M2','M3'),
        ('camera_control_bootstrap.parquet','M2C','M3C')):
        draws=pd.read_parquet(OUT/name)
        for scope in SCOPES:
            sub=draws[draws.scope==scope]
            require(len(sub)==5000 and set(sub.draw)==set(range(5000)),f'Missing bootstrap draws {name} {scope}')
            require((sub.distinct_matches<sub.matches_drawn).any(),
                    f'Match multiplicity lost in {name} {scope}')
            require(sub.control.eq(control).all() and sub.augmented.eq(augmented).all(),
                    f'Wrong bootstrap comparison in {name} {scope}')
    for file,draws_file in (('primary_attribution_validation.csv','primary_360_incremental_bootstrap.parquet'),
                            ('camera_control_validation.csv','camera_control_bootstrap.parquet')):
        table=pd.read_csv(OUT/file);draws=pd.read_parquet(OUT/draws_file)
        require(len(table)==9,'Missing pooled or tournament comparison')
        for row in table.itertuples():
            d=draws[draws.scope==row.scope]
            column={'AUC':'delta_auc','Brier':'delta_brier','log_loss':'delta_log_loss'}[row.metric]
            lo,hi=d[column].quantile([.025,.975])
            require(np.isclose(lo,row.ci_low,atol=1e-12) and np.isclose(hi,row.ci_high,atol=1e-12),
                    f'Bootstrap CI mismatch {file} {row.scope} {row.metric}')
            require(np.isclose(row.delta,row.augmented_value-row.control_value,atol=1e-12),
                    f'Paired metric delta mismatch {file}')
    area=pd.read_csv(OUT/'visible_area_audit.csv')
    require(len(area)==12 and area.pearson_r.notna().all() and area.area_fraction_min.gt(0).all()
            and area.area_fraction_max.le(1).all(),'Visible-area robustness absent')
    camera=pd.read_csv(OUT/'camera_control_validation.csv')
    primary=pd.read_csv(OUT/'primary_attribution_validation.csv')
    require(set(primary.control)=={'M2'} and set(primary.augmented)=={'M3'},
            '360 increment attributed from wrong baseline')
    require(set(camera.control)=={'M2C'} and set(camera.augmented)=={'M3C'},
            'Camera-control attribution wrong')
    ablation=pd.read_csv(OUT/'component_ablation.csv')
    require(set(ablation.component)=={'clear_short_only','central_exit_only','both_core_360'},
            'Component ablation incomplete')

    manifest=pd.read_csv(OUT/'source_manifest_verified.csv')
    require(len(manifest)==348 and manifest.bytes.gt(0).all() and manifest.http_status.eq(200).all()
            and manifest.json_valid.all() and manifest.sha256.notna().all()
            and manifest.official_commit.nunique()==1,'Verified manifest has a zero or invalid source')
    if not archive_mode:
        require(RAW.is_dir(),'Fresh raw directory absent; use --archive for package-only verification')
        for row in manifest.itertuples():
            path=RAW/row.relative_path
            require(path.is_file() and path.stat().st_size==row.bytes and sha256(path)==row.sha256,
                    f'Verified raw hash/size mismatch: {row.relative_path}')
            with path.open('rb') as file:
                json.load(file)
    reproduction=pd.read_csv(OUT/'raw_to_derived_reproduction.csv')
    require(len(reproduction)>0 and reproduction.status.eq('MATCH').all()
            and reproduction.discrepancies.eq(0).all() and
            set(reproduction.table)=={'distribution_option_summary.parquet','option_candidates.parquet'},
            'Unexplained raw-to-derived discrepancy')
    require(reproduction[reproduction.field=='__key__'].frozen_rows.tolist()==[4831,27777],
            'Reproduction row count changed')

    decision=json.loads((OUT/'promotion_decision.json').read_text())
    pooled=primary[primary.scope=='ALL']
    directional=gate_direction(primary,'WC2022') and gate_direction(primary,'EURO2024')
    ci=any(r.ci_high<0 for r in pooled.itertuples() if r.metric in ('Brier','log_loss'))
    auc=float(pooled[pooled.metric=='AUC'].delta.iloc[0])>0
    section_c=directional and ci and auc
    camera_pass=gate_direction(camera,'ALL')
    expected_main=section_c and camera_pass and reproduction.status.eq('MATCH').all()
    require(decision['section_11c_pass']==section_c and decision['camera_control_pooled_improves']==camera_pass,
            'Section 11C or camera gate not correctly applied')
    require((decision['360_incremental_xr_result']=='MAIN')==expected_main,
            'Incremental main-paper claim violates rule')
    require(decision['abstract_360_claim_approved']==expected_main,
            'Abstract attribution claim violates rule')
    report_text=report.read_text().lower()
    require('m4 additionally' in report_text and 'secondary' in report_text,
            'Report treats camera-sensitive M4 as primary')
    require('m3 versus m2' in report_text and 'not causal evidence' in report_text,
            'Attribution or observational limitation absent')
    require('causes the goalkeeper' not in report_text and re.search(r'\bproves\b', report_text) is None,
            'Causal language in report')
    print('VERIFY STATSBOMB360 R1.2B: PASS' + (' (archive evidence mode)' if archive_mode else ' (full raw mode)'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',action='store_true',help='verify packaged evidence without raw provider JSON')
    args=parser.parse_args()
    main(archive_mode=args.archive)
