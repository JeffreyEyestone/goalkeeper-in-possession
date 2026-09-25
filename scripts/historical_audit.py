from pathlib import Path
import json, hashlib, platform
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
claims={'WC2022':4181,'PL1516':23010,'T2024':5405,'WSL2021':10722}
rows=[]
for cohort,n in claims.items():
    p=ROOT/f'historical/goalkeeper-in-possession-repo/out/{cohort}_valued.parquet'; d=pd.read_parquet(p)
    rows.append({'cohort':cohort,'paper_n':n,'artifact_n':len(d),'delta':len(d)-n,'matches':d.match_id.nunique(),'duplicate_actions':int(d.duplicated(['match_id','idx']).sum()),'completion':float(d.complete.mean()),'raw_retention':float(d.retain.mean()),'mean_rho':float(d.rho.mean()),'mean_native_length':float(d.length.mean()),'artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(ROOT/'results/v2_0_reproduction/artifact_audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))

# Recompute legacy band summaries from supplied artifacts, not from fitted models.
bands=[]
for cohort in ['WC2022','T2024','WSL2021']:
    d=pd.read_parquet(ROOT/f'historical/goalkeeper-in-possession-repo/out/{cohort}_valued.parquet');d=d[d.is_gkick].copy()
    for unit,dist in [('native_yards',d.length),('converted_meters',d.length*.9144)]:
        d['band']=pd.cut(dist,[0,25,40,60,float('inf')],right=False,labels=['<25','25-40','40-60','>=60'])
        for b,g in d.groupby('band',observed=True):bands.append({'cohort':cohort,'band_units':unit,'band':b,'n':len(g),'historical_xtgk_mean':g.xtgk.mean()})
pd.DataFrame(bands).to_csv(ROOT/'results/v2_0_reproduction/distance_unit_audit.csv',index=False)
