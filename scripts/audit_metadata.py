from pathlib import Path
import json,subprocess,collections,hashlib
ROOT=Path(__file__).resolve().parents[1]
COHORTS={'WC2018':[(43,3)],'WC2022':[(43,106)],'PL1516':[(2,27)],'WSL2021':[(37,90)],'T2024':[(55,282),(223,282)]}
def main():
    base=json.loads((ROOT/'config/data_source.json').read_text())['base_url'];dest=ROOT/'data/raw/metadata';dest.mkdir(parents=True,exist_ok=True);result=[]
    for cohort,pairs in COHORTS.items():
        matches=[];hashes={}
        for comp,season in pairs:
            p=dest/f'{comp}_{season}.json'
            if not p.exists():p.write_bytes(subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',f'{base}/matches/{comp}/{season}.json'],capture_output=True,check=True).stdout)
            matches+=json.loads(p.read_text());hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        supplied=set(json.loads((ROOT/f'data/manifests/matches_{cohort}.json').read_text()));available={m['match_id'] for m in matches}
        result.append({'cohort':cohort,'frozen_matches':len(supplied),'available_matches':len(available),'frozen_not_in_source':sorted(supplied-available),'source_not_in_frozen':sorted(available-supplied),'data_versions':dict(collections.Counter(m.get('metadata',{}).get('data_version','missing') for m in matches if m['match_id'] in supplied)),'metadata_sha256':hashes})
    (ROOT/'results/cohort_metadata_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
