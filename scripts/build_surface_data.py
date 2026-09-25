from pathlib import Path
import json,sys
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.labels import name,seconds

def main():
    states=[];criteria=[];actions=pd.read_parquet(ROOT/'data/processed/actions.parquet');groups={m:g for m,g in actions.groupby('match_id')}
    for mf in sorted((ROOT/'data/manifests').glob('matches_*.json')):
        cohort=mf.stem.removeprefix('matches_')
        for mid in json.loads(mf.read_text()):
            ev=json.loads((ROOT/f'data/raw/events/{mid}.json').read_text());by={e['id']:e for e in ev};poss={}
            for e in ev:
                key=(e['period'],e['possession']);poss.setdefault(key,[]).append(e)
            poss_xg={k:sum(e.get('shot',{}).get('statsbomb_xg',0) for e in group if name(e)=='Shot' and e['team']['id']==e['possession_team']['id']) for k,group in poss.items()}
            poss_order=list(poss);remaining={}
            for k,events in poss.items():
                rem=0
                for e in reversed(events):
                    if name(e)=='Shot' and e['team']['id']==e['possession_team']['id']:rem+=e['shot'].get('statsbomb_xg',0)
                    remaining[e['id']]=rem
            first_seen=set()
            for e in ev:
                if name(e) not in ('Pass','Carry','Ball Receipt*','Dribble','Shot') or not e.get('location'):continue
                key=(e['period'],e['possession']);start=key not in first_seen;first_seen.add(key);xy=e['location'];end=e.get('pass',e.get('carry',{})).get('end_location',xy)
                states.append({'cohort':cohort,'match_id':mid,'period':e['period'],'possession':e['possession'],'x':xy[0],'y':xy[1],'pressure':int(e.get('under_pressure',False)),'own':e['team']['id']==e['possession_team']['id'],'start':start,'rem':remaining[e['id']],'poss_xg':poss_xg[key],'etype':name(e),'end_x':end[0],'end_y':end[1],'complete':int('outcome' not in e.get('pass',{})),'shot_xg':e.get('shot',{}).get('statsbomb_xg',0)})
            for a in groups[mid].itertuples():
                e=by[a.event_id];start=seconds(e);key=(e['period'],e['possession']);own=e['team']['id'];row={'event_id':a.event_id,'criterion_short':0.,'criterion_possession':remaining[a.event_id] if e['possession_team']['id']==own else 0.}
                # First opponent possession after the action; use that possession's full shot xG.
                for k in poss_order[poss_order.index(key):]:
                    events=poss[k]
                    if k[0]!=e['period']:break
                    if events[0]['possession_team']['id']!=own:
                        row['criterion_possession']-=poss_xg[k];break
                for future in ev[e['index']:]:
                    if future['period']!=e['period'] or seconds(future)-start>10:break
                    if name(future)=='Shot':row['criterion_short']+=(1 if future['team']['id']==own else -1)*future['shot'].get('statsbomb_xg',0)
                for label in ['R1','R1_control','R2','R3','R3_control','R15']:
                    sid=getattr(a,label+'_state_id');s=by.get(sid,{})
                    row[label+'_state_pressure']=int(s.get('under_pressure',False));row[label+'_state_team']=s.get('team',{}).get('id');row[label+'_state_type']=name(s)
                criteria.append(row)
        print(cohort,'surface states/criteria collected',flush=True)
    pd.DataFrame(states).to_parquet(ROOT/'data/processed/surface_states.parquet',index=False)
    pd.DataFrame(criteria).to_parquet(ROOT/'data/processed/criteria.parquet',index=False)
if __name__=='__main__':main()
