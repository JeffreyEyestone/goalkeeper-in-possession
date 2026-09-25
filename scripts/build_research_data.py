from pathlib import Path
import sys,json,collections,math
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gkpossession.labels import study_labels,seconds,name
from gkpossession.targets import candidate_receipt
from gkpossession.features import distance

def main():
    out=ROOT/'data/processed';out.mkdir(exist_ok=True);rows=[];examples={}
    for mf in sorted((ROOT/'data/manifests').glob('matches_*.json')):
        cohort=mf.stem.removeprefix('matches_');n=0
        for mid in json.loads(mf.read_text()):
            events=json.loads((ROOT/f'data/raw/events/{mid}.json').read_text());ids={e['id']:e for e in events};score=collections.Counter()
            for i,e in enumerate(events):
                if name(e)=='Pass' and e.get('position',{}).get('name')=='Goalkeeper':
                    p=e['pass'];loc=e.get('location',[float('nan')]*2);end=p.get('end_location',[float('nan')]*2);rec=candidate_receipt(e,ids);target=rec['location'] if rec else [float('nan')]*2
                    r={'cohort':cohort,'match_id':mid,'event_id':e['id'],'event_index':e['index'],'period':e['period'],'keeper_id':e['player']['id'],'keeper':e['player']['name'],'team_id':e['team']['id'],'team':e['team']['name'],
                       'origin_x':loc[0],'origin_y':loc[1],'origin_pressure':int(e.get('under_pressure',False)),'goal_kick':int(p.get('type',{}).get('name')=='Goal Kick'),
                       'time_minutes':seconds(e)/60,'second_half':int(e['period']>=2),'score_difference':score[e['team']['id']]-sum(v for t,v in score.items() if t!=e['team']['id']),
                       'height':p.get('height',{}).get('name','Unknown'),'body_part':p.get('body_part',{}).get('name','Unknown'),'technique':p.get('technique',{}).get('name','Standard'),'play_pattern':e.get('play_pattern',{}).get('name','Unknown'),
                       'end_x':end[0],'end_y':end[1],'realized_distance_m':float(distance(end[0]-loc[0],end[1]-loc[1])),'native_distance':math.hypot(end[0]-loc[0],end[1]-loc[1]),
                       'target_available':rec is not None,'target_x':target[0],'target_y':target[1],'target_distance_m':float(distance(target[0]-loc[0],target[1]-loc[1])),'target_angle':math.atan2(target[1]-loc[1],target[0]-loc[0]),
                       'outcome':p.get('outcome',{}).get('name','Complete'),'complete':int('outcome' not in p),'raw_possession':e['possession']}
                    labels=study_labels(events,i)
                    for k,l in labels.items():
                        r[k]=l['label'];r[k+'_status']=l['status'];state=ids.get(l['state_id'],{});xy=state.get('location',[None,None]);r[k+'_state_x']=xy[0];r[k+'_state_y']=xy[1];r[k+'_state_id']=l['state_id']
                    rows.append(r);n+=1
                    key=f"{cohort}:{labels['R1']['status']}:{labels['R1']['label']}"
                    if key not in examples:examples[key]={'pass':e,'labels':labels,'following':events[i+1:i+20]}
                if name(e)=='Shot' and e.get('shot',{}).get('outcome',{}).get('name')=='Goal':score[e['team']['id']]+=1
                elif name(e)=='Own Goal For':score[e['team']['id']]+=1
        print(cohort,n,flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'actions.parquet',index=False)
    (ROOT/'results/v21/label_sequence_examples.json').write_text(json.dumps(examples,indent=2))
    stats=[]
    df['realized_distance_band']=pd.cut(df.realized_distance_m,[-1,25,40,60,float('inf')],right=False,labels=['<25','25-40','40-60','>=60'])
    for label in ['R1','R1_control','R2','R3','R15','R3_control']:
        for dimension in ['all','goal_kick','realized_distance_band']:
            groups=df.groupby('cohort') if dimension=='all' else df.groupby(['cohort',dimension],observed=True)
            for key,g in groups:
                if not isinstance(key,tuple):key=(key,)
                stats.append({'cohort':key[0],'label':label,'dimension':dimension,'level':str(key[1]) if len(key)>1 else 'all','n':len(g),'positive':int((g[label]==1).sum()),'negative':int((g[label]==0).sum()),'ambiguous':int((g[label+'_status']=='ambiguous').sum()),'dead_ball':int((g[label+'_status']=='dead_ball').sum()),'censored':int((g[label+'_status']=='censored').sum()),'base_rate':g[label].mean()})
    pd.DataFrame(stats).to_csv(ROOT/'results/v21/retention_label_study.csv',index=False)
    selection=[]
    for dimension in ['complete','outcome','goal_kick','origin_pressure','height','body_part','technique','play_pattern','keeper','team','match_id','second_half']:
        for (c,level),g in df.groupby(['cohort',dimension],dropna=False,observed=True):
            selection.append({'cohort':c,'dimension':dimension,'level':str(level),'n':len(g),'target_available':int(g.target_available.sum()),'availability_rate':g.target_available.mean(),'completion_rate':g.complete.mean(),'R1_rate':g.R1.mean()})
    pd.DataFrame(selection).to_csv(ROOT/'results/v21/target_selection_strata.csv',index=False)
    print(pd.DataFrame(stats).query("dimension=='all'").to_string(index=False))
if __name__=='__main__':main()
