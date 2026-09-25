import hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from gkpossession.targets import candidate_receipt
from download_data import fetch

def test_manifest_integrity():
    config=json.loads((ROOT/'config/data_source.json').read_text());seen=set()
    for name,digest in config['manifests'].items():
        p=ROOT/'data/manifests'/name;ids=json.loads(p.read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        assert len(ids)==len(set(ids));assert not seen.intersection(ids);seen.update(ids)
    assert len(seen)==722

def fixture():
    a={'pass':{'recipient':{'id':7},'end_location':[19,10]},'related_events':['r'],'team':{'id':1},'period':1}
    r={'id':'r','type':{'name':'Ball Receipt*'},'team':{'id':1},'period':1,'player':{'id':7},'location':[50,30]}
    return a,r

def test_target_uses_linked_receipt_never_endpoint():
    a,r=fixture();assert candidate_receipt(a,{'r':r}) is r
    a['pass']['end_location']=[120,80];assert candidate_receipt(a,{'r':r}) is r
    a['related_events']=[];assert candidate_receipt(a,{'r':r}) is None

@pytest.mark.parametrize('field,value',[('period',2),('team',{'id':2}),('player',{'id':8})])
def test_target_rejects_mismatched_identity(field,value):
    a,r=fixture();r[field]=value;assert candidate_receipt(a,{'r':r}) is None

def test_target_rejects_ambiguous_receipts():
    a,r=fixture();a['related_events']=['r','r2'];assert candidate_receipt(a,{'r':r,'r2':dict(r,id='r2')}) is None

def test_download_empty_cache_and_reuse(tmp_path,monkeypatch):
    class Response: stdout=b'[{"id":"abc"}]'
    calls=[]
    def run(*args,**kwargs):calls.append(args);return Response()
    monkeypatch.setattr('download_data.subprocess.run',run)
    row=fetch(1,'https://example.invalid',tmp_path);assert row['events']==1
    assert fetch(1,'https://example.invalid',tmp_path)==row;assert len(calls)==1

def test_bad_download_does_not_commit_cache(tmp_path,monkeypatch):
    class Response: stdout=b'{}'
    monkeypatch.setattr('download_data.subprocess.run',lambda *a,**kw:Response())
    monkeypatch.setattr('download_data.time.sleep',lambda _:None)
    with pytest.raises(ValueError): fetch(1,'https://example.invalid',tmp_path)
    assert not (tmp_path/'1.json').exists()

def test_cached_data_tampering_fails(tmp_path,monkeypatch):
    (tmp_path/'1.json').write_text('[{"id":"changed"}]')
    monkeypatch.setattr('download_data.time.sleep',lambda _:None)
    with pytest.raises(ValueError,match='Checksum mismatch'):
        fetch(1,'https://example.invalid',tmp_path,'0'*64)

def test_coverage_matches_outcome_totals():
    import csv
    coverage=list(csv.DictReader((ROOT/'results/target_coverage_by_cohort.csv').open()))
    rows=list(csv.DictReader((ROOT/'results/semantics_by_outcome.csv').open()))
    for c in coverage:
        sub=[r for r in rows if r['cohort']==c['cohort']]
        assert sum(int(r['n']) for r in sub)==int(c['eligible_gk_passes'])
        assert sum(int(r['candidate_target']) for r in sub)==int(c['candidate_targets'])

def test_scientific_gate_cannot_report_success():
    import subprocess
    r=subprocess.run([sys.executable,str(ROOT/'scripts/scientific_gate.py')],capture_output=True,text=True)
    assert r.returncode!=0 and ('external_calibration' in r.stderr or 'run make xr' in r.stderr)
