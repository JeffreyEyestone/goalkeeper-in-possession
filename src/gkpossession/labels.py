"""Event-control labels. Outcome/sequence access is confined to targets, never features."""
import math

def seconds(e):
    h,m,s=e['timestamp'].split(':'); return int(h)*3600+int(m)*60+float(s)

def name(e): return e.get('type',{}).get('name','')

def is_control(e):
    t=name(e)
    if not e.get('location'):return False
    if t=='Ball Receipt*':return e.get('ball_receipt',{}).get('outcome',{}).get('name')!='Incomplete'
    if t in ('Carry','Shot'):return True
    if t=='Pass':return e.get('pass',{}).get('body_part',{}).get('name')!='No Touch'
    if t=='Ball Recovery':return not e.get('ball_recovery',{}).get('recovery_failure',False)
    if t=='Dribble':return e.get('dribble',{}).get('outcome',{}).get('name')=='Complete'
    if t=='Interception':return e.get('interception',{}).get('outcome',{}).get('name') in ('Success In Play','Won')
    if t=='Goal Keeper':return e.get('goalkeeper',{}).get('type',{}).get('name') in ('Collected','Keeper Sweeper') and e.get('goalkeeper',{}).get('outcome',{}).get('name') in ('Success','Success In Play','Claim')
    return False

def is_dead(e):
    t=name(e)
    if t in ('Half End','Offside','Injury Stoppage','Referee Ball-Drop'):return True
    if t=='Foul Committed':return not e.get('foul_committed',{}).get('advantage',False)
    if t=='Foul Won':return not e.get('foul_won',{}).get('advantage',False)
    if e.get('out',False):return True
    if t=='Pass':
        p=e['pass'];return p.get('outcome',{}).get('name') in ('Out','Pass Offside','Injury Clearance') or p.get('type',{}).get('name') in ('Throw-in','Free Kick','Corner','Goal Kick','Kick Off')
    return False

def study_labels(events, i):
    a=events[i];start=seconds(a);team=a['team']['id'];period=a['period'];p=a['pass']
    outcome=p.get('outcome',{}).get('name','Complete')
    future=[]
    for e in events[i+1:]:
        if e['period']!=period:break
        dt=seconds(e)-start
        if dt>15:break
        if dt>=0:future.append(e)
    last_time=next((seconds(e)-start for e in reversed(events) if e['period']==period),0)
    # Do not call a restart immediate control of the original live-ball distribution.
    direct_dead=outcome in ('Out','Pass Offside','Injury Clearance')
    result={}
    if direct_dead: result['R1']={'label':None,'status':'dead_ball','state_id':None}
    else:
        result['R1']={'label':None,'status':'ambiguous','state_id':None}
        for e in future:
            restart=name(e)=='Pass' and e.get('pass',{}).get('type',{}).get('name') in ('Throw-in','Free Kick','Corner','Goal Kick','Kick Off')
            if restart:result['R1']['status']='dead_ball';break
            # A deliberate live-ball pass establishes control at its START even if it later goes out.
            if is_control(e):
                result['R1']={'label':int(e['team']['id']==team),'status':'resolved','state_id':e['id']};break
            if is_dead(e):result['R1']['status']='dead_ball';break
    result['R1_control']=dict(result['R1'])
    # Dead-ball resolution: use only an explicitly annotated first restart, not its later outcome.
    # This extends control to possession rights at the immediate distribution resolution.
    if result['R1']['status']=='dead_ball':
        for e in events[i+1:]:
            if e['period']!=period or seconds(e)-start>120:break
            restart=(name(e)=='Pass' and e.get('pass',{}).get('type',{}).get('name') in ('Throw-in','Free Kick','Corner','Goal Kick','Kick Off')) or (name(e)=='Shot' and e.get('shot',{}).get('type',{}).get('name')=='Penalty')
            if restart:
                result['R1']={'label':int(e['team']['id']==team),'status':'restart_rights','state_id':e['id']};break
            # If live control intervenes before an explicit restart, the dead-ball sequence is unresolved.
            if is_control(e):break
    for key,horizon in [('R2',5),('R3',10),('R15',15)]:
        window=[e for e in future if seconds(e)-start<=horizon]
        # Preserve legacy R3 separately below; revised horizon labels require observable control.
        state=None;dead=direct_dead;shot=None;uncertain=False
        for e in window:
            if dead:break
            if name(e)=='Shot' and e['team']['id']==team:shot=e;break
            if is_dead(e):dead=True;break
            if is_control(e):state=e;uncertain=False
            elif name(e) in ('Miscontrol','Dispossessed','Clearance','Duel','50/50','Block'):uncertain=True
        if shot:r={'label':1,'status':'shot','state_id':shot['id']}
        elif dead:r={'label':None,'status':'dead_ball','state_id':None}
        elif last_time<horizon:r={'label':None,'status':'censored','state_id':None}
        elif state is None or uncertain:r={'label':None,'status':'ambiguous','state_id':None}
        else:r={'label':int(state['team']['id']==team),'status':'resolved','state_id':state['id']}
        result[key]=r
    # Exact legacy candidate: last filtered event within 10 sec or own shot; empty=success.
    legacy=[e for e in future if 0<seconds(e)-start<=10 and name(e) in ('Pass','Carry','Ball Receipt*','Dribble','Shot') and e.get('location')]
    result['R3_control']=result['R3']
    result['R3']={'label':int(not legacy or any(name(e)=='Shot' and e['team']['id']==team for e in legacy) or legacy[-1]['possession_team']['id']==team),'status':'legacy_empty_success' if not legacy else 'legacy','state_id':legacy[-1]['id'] if legacy else None}
    return result
