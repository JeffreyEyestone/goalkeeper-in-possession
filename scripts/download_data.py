"""Fetch frozen public event files atomically; hash cached/downloaded bytes."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, time, subprocess
ROOT = Path(__file__).resolve().parents[1]
def fetch(mid, base, dest, expected_sha256=None):
    path = dest / f'{mid}.json'
    for attempt in range(4):
        try:
            raw = path.read_bytes() if path.exists() else subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '60', f'{base}/events/{mid}.json'], check=True, capture_output=True).stdout
            if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise ValueError(f'Checksum mismatch: {mid}; quarantine cache and investigate source revision')
            events = json.loads(raw)
            if not isinstance(events, list) or not events or not all('id' in e for e in events):
                raise ValueError(f'Invalid events: {mid}')
            if not path.exists():
                tmp = path.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(path)
            return {'match_id': mid, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'events': len(events)}
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)
def main():
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int); p.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw/events'); a=p.parse_args()
    config=json.loads((ROOT/'config/data_source.json').read_text())
    ids=set()
    for path in sorted((ROOT/'data/manifests').glob('*.json')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['manifests'][path.name]: raise ValueError('Manifest changed')
        cohort=json.loads(path.read_text()); ids.update(cohort[:a.limit] if a.limit else cohort)
    a.raw_dir.mkdir(parents=True, exist_ok=True); rows=[]
    checksum_path=ROOT/'results/raw_data_checksums.json'
    expected={r['match_id']:r['sha256'] for r in json.loads(checksum_path.read_text())['files']} if checksum_path.exists() else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(fetch, mid, config['base_url'], a.raw_dir, expected.get(mid)) for mid in sorted(ids)]
        for i,job in enumerate(concurrent.futures.as_completed(jobs),1):
            rows.append(job.result())
            if i%50==0 or i==len(ids): print(f'{i}/{len(ids)} verified', flush=True)
    out={'revision':config['revision'],'files':sorted(rows,key=lambda x:x['match_id'])}
    (a.raw_dir.parent/'raw_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
