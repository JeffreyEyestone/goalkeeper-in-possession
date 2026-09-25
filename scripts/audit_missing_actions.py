"""Optional legacy join diagnostic after reproduce_legacy_parse.py."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];rows=[]
for cohort in ['WC2022','T2024','WSL2021','PL1516']:
    d=pd.read_parquet(ROOT/f'data/interim/legacy_parse/data/{cohort}.parquet')
    a=pd.read_parquet(ROOT/f'historical/goalkeeper-in-possession-repo/out/{cohort}_valued.parquet')
    g=d[(d.etype=='Pass')&(d.position=='Goalkeeper')];keys=set(zip(a.match_id,a.idx))
    for _,x in g.iterrows():
        if (x.match_id,x.idx) in keys:continue
        rec=d[(d.match_id==x.match_id)&(d.etype=='Ball Receipt*')&(d.player==x.recipient)]
        near=rec[(rec.idx>x.idx)&(rec.idx<=x.idx+6)]
        rows.append({'cohort':cohort,'match_id':int(x.match_id),'event_index':int(x.idx),'recipient':x.recipient,'same_recipient_receipts_in_match':len(rec),'receipts_next_six_indices':len(near),'consistent_with_legacy_join_exclusion':len(rec)>0 and len(near)==0})
pd.DataFrame(rows).to_csv(ROOT/'results/v2_0_reproduction/missing_valued_actions.csv',index=False)
