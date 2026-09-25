"""Census of frozen raw GK Pass events. No model fitting or paper results."""
from pathlib import Path
import collections,csv,hashlib,json,math,sys,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.targets import candidate_receipt

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    rows=[];examples={};schema=[];coverage=[];source=json.loads((ROOT/'config/data_source.json').read_text())
    for mf in sorted((ROOT/'data/manifests').glob('*.json')):
        cohort=mf.stem.removeprefix('matches_');counts=collections.defaultdict(collections.Counter);keys=collections.Counter();other=collections.Counter()
        for mid in json.loads(mf.read_text()):
            events=json.loads((ROOT/f'data/raw/events/{mid}.json').read_text());by_id={e['id']:e for e in events}
            for e in events:
                if e.get('position',{}).get('name')!='Goalkeeper':continue
                other[e['type']['name']]+=1
                if e['type']['name']!='Pass':continue
                p=e['pass'];out=p.get('outcome',{}).get('name','Complete');c=counts[out];c['n']+=1
                keys.update(p.keys());c['recipient_present']+=bool(p.get('recipient'));c['end_present']+=len(p.get('end_location',[]))>=2;c['origin_present']+=len(e.get('location',[]))>=2
                c['under_pressure_present']+='under_pressure' in e;c['under_pressure_true']+=e.get('under_pressure') is True;c['keeper_arm']+=p.get('body_part',{}).get('name')=='Keeper Arm';c['goal_kick']+=p.get('type',{}).get('name')=='Goal Kick'
                receipts=[by_id[r] for r in e.get('related_events',[]) if r in by_id and by_id[r]['type']['name']=='Ball Receipt*']
                c['related_receipt_any']+=bool(receipts);rec=candidate_receipt(e,by_id);c['candidate_target']+=rec is not None
                if rec:
                    c['receipt_incomplete']+=rec.get('ball_receipt',{}).get('outcome',{}).get('name')=='Incomplete'
                    c['receipt_pressure_present']+='under_pressure' in rec
                    end=p.get('end_location');loc=rec['location']
                    c['target_differs_end_gt1']+=bool(end and math.dist(end[:2],loc[:2])>1)
                if p.get('end_location') and e.get('location'):
                    dx=p['end_location'][0]-e['location'][0];dy=p['end_location'][1]-e['location'][1]
                    if p.get('length') is not None:
                        c['length_available']+=1;c['length_matches_endpoint_0_01']+=abs(math.hypot(dx,dy)-p['length'])<.01
                    if p.get('angle') is not None:
                        c['angle_available']+=1;c['angle_matches_endpoint_0_001']+=abs(math.atan2(dy,dx)-p['angle'])<.001
                related=[by_id[r] for r in e.get('related_events',[]) if r in by_id]
                categories=[out]
                if any(x['type']['name']=='Interception' for x in related):categories+=['intercepted']
                if any(x['type']['name']=='Block' for x in related):categories+=['blocked']
                if any(x['type']['name'] in ('Duel','50/50') for x in events[max(0,e['index']-1):e['index']+8]):categories+=['contested_nearby']
                for category in categories:
                    key=cohort+':'+category
                    if key not in examples:
                        examples[key]={'cohort':cohort,'match_id':mid,'category':category,'pass':e,'related_events':related,'next_events':events[max(0,e['index']):e['index']+8]}
        fields=['n','recipient_present','end_present','origin_present','under_pressure_present','under_pressure_true','keeper_arm','goal_kick','related_receipt_any','candidate_target','receipt_incomplete','receipt_pressure_present','target_differs_end_gt1','length_available','length_matches_endpoint_0_01','angle_available','angle_matches_endpoint_0_001']
        for out,c in counts.items():rows.append({'cohort':cohort,'outcome':out,**{k:c[k] for k in fields}})
        total=sum(c['n'] for c in counts.values());target=sum(c['candidate_target'] for c in counts.values());failed=sum(c['n'] for o,c in counts.items() if o!='Complete');failed_target=sum(c['candidate_target'] for o,c in counts.items() if o!='Complete')
        coverage.append({'cohort':cohort,'matches':len(json.loads(mf.read_text())),'eligible_gk_passes':total,'candidate_targets':target,'missing_targets':total-target,'missing_fraction':(total-target)/total,'failed_passes':failed,'failed_candidate_targets':failed_target,'failure_missing_fraction':(failed-failed_target)/failed,'gate_gt_15_percent':(total-target)/total>.15})
        schema.append({'cohort':cohort,'pass_field_counts':dict(keys),'gk_event_types':dict(other)})
        print(cohort,coverage[-1],flush=True)
    write_csv(ROOT/'results/target_coverage_by_cohort.csv',coverage);write_csv(ROOT/'results/semantics_by_outcome.csv',rows)
    (ROOT/'results/semantics_examples.json').write_text(json.dumps(examples,indent=2));(ROOT/'results/schema_by_cohort.json').write_text(json.dumps(schema,indent=2))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance={'specification_id':'scientific-v2.1-pre-model-audit','git_commit':commit,'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':source['revision'],'manifest_hashes':source['manifests'],'raw_manifest_sha256':hashlib.sha256((ROOT/'data/raw/raw_manifest.json').read_bytes()).hexdigest(),'code_function':'scripts/audit_semantics.py:main','status':'audit_only_not_paper_results'}
    (ROOT/'results/audit_provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__': main()
