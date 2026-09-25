"""Reproducible R1.2 StatsBomb 360 visible-option pilot (official open data)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

RAW=Path('/private/tmp/statsbomb360_raw'); OUT=Path('results/statsbomb360'); FIG=Path('figures/statsbomb360'); REP=Path('reports')
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); (FIG/'cases').mkdir(exist_ok=True)

def loc(x): return x if isinstance(x,list) and len(x)>=2 else [np.nan,np.nan]
def std(p): return (p[0]*105/120,p[1]*68/80)
def family(d): return 'SHORT' if d<25 else ('MEDIUM' if d<40 else 'DIRECT')
def side(y): return 'LEFT' if y<22.67 else ('CENTRAL' if y<=45.33 else 'RIGHT')
def segdist(a,b,p):
    ax,ay=a; bx,by=b; px,py=p; vx,vy=bx-ax,by-ay; den=vx*vx+vy*vy
    if den==0:return math.dist(a,p),False
    t=((px-ax)*vx+(py-ay)*vy)/den
    q=(ax+t*vx,ay+t*vy); return math.dist(q,p),0<=t<=1
def boot_ci(df, metric, reps=5000, seed=7):
    rng=np.random.default_rng(seed); matches=df.match_id.unique(); out=[]
    for _ in range(reps):
        ids=rng.choice(matches,len(matches),replace=True); s=pd.concat([df[df.match_id==i] for i in ids])
        out.append(metric(s))
    return np.quantile(out,[.025,.975]),np.array(out)
def glm_table(df,y,cols,name):
    X=sm.add_constant(df[cols].astype(float),has_constant='add'); m=sm.GLM(df[y].astype(float),X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':df.match_id})
    z=1.96; ci=m.conf_int(); t=pd.DataFrame({'term':m.params.index,'coefficient':m.params.values,'odds_ratio':np.exp(m.params.values),'ci_low':np.exp(ci[0].values),'ci_high':np.exp(ci[1].values),'p_value':m.pvalues.values,'cluster':'match','n':len(df),'outcome':name})
    return t,m
def plot_pitch(ax):
    ax.plot([0,105,105,0,0],[0,0,68,68,0],c='#202020'); ax.axvline(52.5,c='#ddd'); ax.set(xlim=(-2,107),ylim=(-2,70),aspect='equal');ax.axis('off')

def main():
    selected=json.loads((RAW/'selected_matches.json').read_text()); frames={}
    for p in (RAW/'three-sixty').glob('*.json'):
        for f in json.loads(p.read_text()): frames[f['event_uuid']]=f
    events=[]; audit=[]; cov=[]; candidates=[]; summaries=[]; frameaudit=[]
    for label,m in selected:
        mid=m['match_id']; ev=json.loads((RAW/'events'/f'{mid}.json').read_text()); lineup=json.loads((RAW/'lineups'/f'{mid}.json').read_text())
        keeper_ids={pl['player_id'] for team in lineup for pl in team.get('lineup',[]) if any(pos.get('position','')=='Goalkeeper' for pos in pl.get('positions',[]))}
        eligible=[]
        for e in ev:
            if e.get('type',{}).get('name')!='Pass': continue
            isgk=e.get('player',{}).get('id') in keeper_ids or e.get('position',{}).get('name')=='Goalkeeper'
            if not isgk: continue
            pp=e.get('pass',{}); start=loc(e.get('location')); end=loc(pp.get('end_location'))
            if np.isnan(start[0]) or np.isnan(end[0]): continue
            d=math.dist(std(start),std(end)); goal=e.get('play_pattern',{}).get('name')=='From Goal Kick'
            rec=pp.get('recipient',{}).get('id')
            row={'event_id':e['id'],'match_id':mid,'competition':label,'keeper_id':e.get('player',{}).get('id'),'keeper_name':e.get('player',{}).get('name'),'team_id':e.get('team',{}).get('id'),'goal_kick':int(goal),'restart_type':'GOAL KICK' if goal else 'OPEN PLAY / RECYCLED POSSESSION','under_pressure':int(e.get('under_pressure',False)),'retained':int('outcome' not in pp),'choice_family':family(d),'chosen_distance_m':d,'chosen_receiver_id':rec,'start_x':std(start)[0],'start_y':std(start)[1],'end_x':std(end)[0],'end_y':std(end)[1]}
            eligible.append((e,row)); events.append(row)
            audit.append({'event_type':'Pass','play_pattern':e.get('play_pattern',{}).get('name'),'included':'YES','reason':'Goalkeeper pass identified by lineup/player position','n':1})
        n360=sum(e['id'] in frames for e,_ in eligible)
        cov.append({'competition':label,'match_id':mid,'restart':'ALL','matches_total':1,'eligible_gk_distributions':len(eligible),'gk_distributions_with_360_frame':n360,'coverage_pct':100*n360/max(1,len(eligible))})
        for e,row in eligible:
            f=frames.get(e['id']); frameaudit.append({'event_id':e['id'],'match_id':mid,'frame_found':bool(f),'actor_is_in_frame':bool(f and any(x.get('actor') for x in f['freeze_frame'])),'event_actor_is_gk':True,'method':'event actor and lineup goalkeeper identity; start location retained'})
            if not f: continue
            a=(row['start_x'],row['start_y']); ff=f['freeze_frame']; team=[x for x in ff if x.get('teammate') and not x.get('actor')]
            opp=[x for x in ff if not x.get('teammate')]
            options=[]
            for j,x in enumerate(team):
                b=std(loc(x.get('location'))); d=math.dist(a,b); near=[math.dist(b,std(loc(o.get('location')))) for o in opp]
                lane=[]
                for th in (1,1.5,2): lane.append(not any(segdist(a,b,std(loc(o.get('location'))))[1] and segdist(a,b,std(loc(o.get('location'))))[0]<=th for o in opp))
                options.append({'event_id':e['id'],'match_id':mid,'competition':label,'candidate_index':j,'candidate_player_id':x.get('player',{}).get('id'),'candidate_name':x.get('player',{}).get('name'),'candidate_x_m':b[0],'candidate_y_m':b[1],'distance_m':d,'forward_progress_m':b[0]-a[0],'lateral_displacement_m':b[1]-a[1],'side':side(b[1]),'distance_family':family(d),'receiver_nearest_opponent_m':min(near) if near else np.nan,'opponents_within_3m':sum(v<=3 for v in near),'opponents_within_5m':sum(v<=5 for v in near),'lane_clear_1m':lane[0],'lane_clear_1_5m':lane[1],'lane_clear_2m':lane[2],'visible_area_present':bool(f.get('visible_area'))})
            candidates.extend(options)
            short=[o for o in options if o['distance_m']<25]; clear=[o for o in short if o['lane_clear_1_5m']]; central=[o for o in clear if o['side']=='CENTRAL']; build=[o for o in options if o['side']=='CENTRAL' and 8<=o['distance_m']<=35 and o['forward_progress_m']>=5 and o['lane_clear_1_5m']]
            summaries.append({**row,'receiver_option_count':len(options),'visible_short_options':len(short),'lane_clear_short_options':len(clear),'central_short_exit':int(bool(central)),'central_build_exit':int(bool(build)),'frame_visible_area_present':bool(f.get('visible_area')),'frame_json':json.dumps(f)})
    evdf=pd.DataFrame(events); opt=pd.DataFrame(candidates); s=pd.DataFrame(summaries)
    opt.to_parquet(OUT/'option_candidates.parquet',index=False); s.to_parquet(OUT/'distribution_option_summary.parquet',index=False)
    pd.DataFrame(audit).groupby(['event_type','play_pattern','included','reason'],dropna=False).n.sum().reset_index().to_csv(OUT/'gk_event_definition_audit.csv',index=False)
    pd.DataFrame(cov).to_csv(OUT/'coverage_audit.csv',index=False); pd.DataFrame(frameaudit).to_csv(OUT/'gk_frame_link_audit.csv',index=False)
    s['clear_group']=pd.cut(s.lane_clear_short_options,[-1,0,1,np.inf],labels=['0','1','2+']); s['pressure_state']=np.where(s.under_pressure==1,'UNDER PRESSURE','FREE')
    choice=s.groupby('clear_group',observed=False).agg(n=('event_id','size'),short_pct=('choice_family',lambda x:100*(x=='SHORT').mean()),medium_pct=('choice_family',lambda x:100*(x=='MEDIUM').mean()),direct_pct=('choice_family',lambda x:100*(x=='DIRECT').mean())).reset_index(); choice.to_csv(OUT/'choice_by_option_count.csv',index=False)
    cen=s.groupby('central_short_exit').agg(n=('event_id','size'),central_choice_pct=('choice_family',lambda x:100*(x=='SHORT').mean()),short_pct=('choice_family',lambda x:100*(x=='SHORT').mean()),medium_pct=('choice_family',lambda x:100*(x=='MEDIUM').mean()),direct_pct=('choice_family',lambda x:100*(x=='DIRECT').mean()),retention_pct=('retained',lambda x:100*x.mean())).reset_index();cen.to_csv(OUT/'central_exit_analysis.csv',index=False)
    s[s.choice_family=='DIRECT'].groupby('clear_group',observed=False).size().rename('n').reset_index().assign(percent=lambda x:100*x.n/x.n.sum()).to_csv(OUT/'direct_despite_short_option.csv',index=False)
    s['direct']=(s.choice_family=='DIRECT').astype(int); s['comp_euro']=(s.competition=='EURO2024').astype(int)
    cols=['lane_clear_short_options','central_short_exit','under_pressure','goal_kick','comp_euro']; dt,dm=glm_table(s,'direct',cols,'DIRECT');dt.to_csv(OUT/'direct_choice_regression.csv',index=False); rt,rm=glm_table(s,'retained',['lane_clear_short_options','direct','under_pressure','goal_kick','comp_euro'],'RETAINED');rt.to_csv(OUT/'retention_regression.csv',index=False)
    # Frozen accepted xR is only joined by exact ID; never fitted here.
    oof=pd.read_parquet('results/winner_gate13/oof_action_values.parquet')[['event_id','rho_oof','R1']]; j=s.merge(oof,on='event_id',how='left'); pd.DataFrame([{'all_360_actions':len(s),'exact_frozen_xr_joined':j.rho_oof.notna().sum(),'join_pct':100*j.rho_oof.notna().mean(),'method':'exact event UUID'}]).to_csv(OUT/'frozen_xr_join_audit.csv',index=False)
    jj=j.dropna(subset=['rho_oof']).copy(); vals=[]; boots=pd.DataFrame()
    if len(jj)>=20 and jj.match_id.nunique()>=5 and jj.R1.nunique()>1:
        jj['rho_clip']=jj.rho_oof.clip(.001,.999); gkf=GroupKFold(5); aug=np.zeros(len(jj))
        feats=['rho_clip','lane_clear_short_options','central_short_exit','receiver_option_count','under_pressure','goal_kick','comp_euro']
        for tr,te in gkf.split(jj,groups=jj.match_id):
            X=sm.add_constant(jj.iloc[tr][feats]); model=sm.GLM(jj.iloc[tr].R1,X,family=sm.families.Binomial()).fit(); aug[te]=model.predict(sm.add_constant(jj.iloc[te][feats],has_constant='add'))
        jj['augmented_oof']=aug
        def mets(d): return [roc_auc_score(d.R1,d.rho_clip),brier_score_loss(d.R1,d.rho_clip),log_loss(d.R1,d.rho_clip),roc_auc_score(d.R1,d.augmented_oof),brier_score_loss(d.R1,d.augmented_oof),log_loss(d.R1,d.augmented_oof)]
        a=mets(jj); vals=[{'metric':k,'frozen_xr':a[i],'frozen_xr_plus_360':a[i+3],'delta_augmented_minus_frozen':a[i+3]-a[i],'n':len(jj),'matches':jj.match_id.nunique(),'evaluation':'5-fold GroupKFold by match; same exact IDs'} for i,k in enumerate(['AUC','Brier','log_loss'])]
        rng=np.random.default_rng(12); ms=jj.match_id.unique(); rows=[]
        for b in range(5000):
            d=pd.concat([jj[jj.match_id==m] for m in rng.choice(ms,len(ms),replace=True)]); z=mets(d);rows.append({'draw':b,'delta_auc':z[3]-z[0],'delta_brier':z[4]-z[1],'delta_log_loss':z[5]-z[2]})
        boots=pd.DataFrame(rows);boots.to_parquet(OUT/'xr_incremental_bootstrap.parquet',index=False)
        for v,col in zip(vals,['delta_auc','delta_brier','delta_log_loss']): v['ci_low'],v['ci_high']=boots[col].quantile([.025,.975]).values
    else: vals=[{'metric':'NOT_EVALUABLE','frozen_xr':np.nan,'frozen_xr_plus_360':np.nan,'delta_augmented_minus_frozen':np.nan,'n':len(jj),'matches':jj.match_id.nunique(),'evaluation':'Insufficient exact frozen-xR join for 5 match-grouped folds'}]; pd.DataFrame({'draw':[]}).to_parquet(OUT/'xr_incremental_bootstrap.parquet',index=False)
    pd.DataFrame(vals).to_csv(OUT/'xr_incremental_validation.csv',index=False)
    cal=jj.assign(clear_group=pd.cut(jj.lane_clear_short_options,[-1,0,1,np.inf],labels=['0','1','2+'])).groupby('clear_group',observed=False).agg(n=('R1','size'),mean_rho=('rho_oof','mean'),actual_retention=('R1','mean')).reset_index();cal['R1_minus_rho']=cal.actual_retention-cal.mean_rho;cal.to_csv(OUT/'xr_option_calibration.csv',index=False)
    role=pd.DataFrame({'definition':['SPATIAL CENTRAL EXIT','ROLE-CONFIRMED CENTRAL MIDFIELD EXIT'],'visible_n':[s.central_build_exit.sum(),np.nan],'lane_clear_n':[s.central_short_exit.sum(),np.nan],'chosen_n':[np.nan,np.nan],'chosen_pct':[np.nan,np.nan],'retained_when_chosen':[np.nan,np.nan],'direct_share_when_visible':[s.loc[s.central_build_exit==1,'direct'].mean(),np.nan],'direct_share_when_absent':[s.loc[s.central_build_exit==0,'direct'].mean(),np.nan],'note':['Static spatial snapshot; no role/movement inference','Lineup event-time role spans unavailable in provider data; not classified']});role.to_csv(OUT/'central_midfield_role_analysis.csv',index=False)
    # Coverage selection comparison is descriptive and uses eligible GK passes.
    evdf['covered']=evdf.event_id.isin(s.event_id); sel=evdf.groupby(['competition','covered']).agg(n=('event_id','size'),goal_kick_share=('goal_kick','mean'),pressure_share=('under_pressure','mean'),retention_share=('retained','mean'),direct_share=('choice_family',lambda x:(x=='DIRECT').mean())).reset_index();sel.to_csv(OUT/'coverage_selection_audit.csv',index=False)
    prof=s.groupby(['competition','keeper_name']).agg(n=('event_id','size'),short_share=('choice_family',lambda x:(x=='SHORT').mean()),medium_share=('choice_family',lambda x:(x=='MEDIUM').mean()),direct_share=('choice_family',lambda x:(x=='DIRECT').mean()),mean_clear_short=('lane_clear_short_options','mean'),central_exit_frequency=('central_short_exit','mean'),direct_despite_clear=('direct',lambda x:np.nan),retention=('retained','mean')).reset_index();prof=prof[prof.n>=30];prof.to_csv(OUT/'keeper_descriptive_profiles.csv',index=False)
    facts=[{'statement':'Visible lane-clear short-option counts are descriptively associated with choice in the 360-covered sample.','n_actions':len(s),'n_matches':s.match_id.nunique(),'competitions':'WC2022, EURO2024','estimate':'See choice_by_option_count.csv','uncertainty':'Descriptive; event-linked snapshot coverage','status':'EXPLORATORY','recommend_for_abstract':'NO','reason':'Pilot selection and static-frame limitations; no main-paper eligibility decision.'}]
    (OUT/'abstract_candidate_facts.json').write_text(json.dumps(facts,indent=2))
    # Figures
    plt.style.use('seaborn-v0_8-whitegrid'); fig,ax=plt.subplots(figsize=(7,4));ax.bar(choice.clear_group,choice.direct_pct,color='#1f4e79');ax.set(ylabel='Direct share (%)',xlabel='Visible lane-clear short options');[ax.text(i,v+1,f'n={int(choice.n.iloc[i])}',ha='center') for i,v in enumerate(choice.direct_pct)];fig.tight_layout();fig.savefig(FIG/'direct_share_by_clear_options.png',dpi=180);plt.close(fig)
    mat=s.groupby(['clear_group','pressure_state','choice_family'],observed=False).size().rename('n').reset_index();fig,ax=plt.subplots(figsize=(8,4));pd.crosstab([s.clear_group,s.pressure_state],s.choice_family,normalize='index').reindex(columns=['SHORT','MEDIUM','DIRECT']).plot.bar(stacked=True,ax=ax);ax.set(ylabel='Choice share',xlabel='clear options × event pressure');fig.tight_layout();fig.savefig(FIG/'choice_option_pressure_matrix.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));cen.set_index('central_short_exit')[['short_pct','medium_pct','direct_pct']].plot.bar(stacked=True,ax=ax);ax.set(ylabel='Choice (%)',xlabel='Visible central short exit');fig.tight_layout();fig.savefig(FIG/'central_exit_choice.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));vv=pd.DataFrame(vals);ax.bar(vv.metric,vv.delta_augmented_minus_frozen.fillna(0),color='#6a8f3d');ax.axhline(0,c='black');ax.set(ylabel='Augmented − frozen xR');fig.tight_layout();fig.savefig(FIG/'xr_incremental_information.png',dpi=180);plt.close(fig)
    # deterministic event cases and a main representative
    rules=[('pressure_short',(s.under_pressure==1)&(s.choice_family=='SHORT')),('pressure_direct',(s.under_pressure==1)&(s.choice_family=='DIRECT')),('central_short',(s.central_short_exit==1)&(s.choice_family=='SHORT')),('central_noncentral_direct',(s.central_short_exit==1)&(s.choice_family=='DIRECT'))]
    examples=[]
    for label,mask in rules:
        for _,r in s[mask].sort_values('event_id').head(3).iterrows():
            f=json.loads(r.frame_json); fig,ax=plt.subplots(figsize=(8,5));plot_pitch(ax); va=np.array(f.get('visible_area',[]));
            if len(va)>2: ax.add_patch(Polygon(np.array([std(q) for q in va]),closed=True,color='#cfe8f3',alpha=.35))
            for x in f['freeze_frame']:
                q=std(loc(x.get('location'))); col='#26734d' if x.get('teammate') else '#b33a3a'; ax.scatter(*q,c=col,s=35)
            ax.scatter(r.start_x,r.start_y,c='gold',edgecolors='black',s=90,zorder=4);ax.arrow(r.start_x,r.start_y,r.end_x-r.start_x,r.end_y-r.start_y,width=.15,color='black',length_includes_head=True)
            ax.set_title(f"{r.competition} | match {r.match_id} | event {r.event_id}\n{r.choice_family}; clear short={r.lane_clear_short_options}; central exit={'YES' if r.central_short_exit else 'NO'}; pressure={r.under_pressure}",fontsize=8)
            fn=FIG/'cases'/f'{label}_{r.event_id}.png';fig.tight_layout();fig.savefig(fn,dpi=160);plt.close(fig);examples.append(fn)
    if examples: (FIG/'main_option_set_example.png').write_bytes(examples[0].read_bytes())
    # reports avoid causal language and no claims about off-camera players.
    coverage=100*len(s)/len(evdf); main_ok=len(s)>=500 and s.match_id.nunique()>=20 and set(s.competition)=={'WC2022','EURO2024'} and s.groupby('match_id').size().max()/len(s)<=.1
    (REP/'STATSBOMB360_COACH_FINDINGS.md').write_text(f"# StatsBomb 360 coach pilot\n\nThe sample contains {len(s)} event-linked goalkeeper pass frames across {s.match_id.nunique()} matches ({coverage:.1f}% of eligible goalkeeper distributions). These snapshots show visible players and lane geometry at the event only; they do not show movement or off-camera options.\n\n## When can we play short?\n\nSee `choice_by_option_count.csv` for observed short, medium, and direct shares by visible lane-clear short options.\n\n## Central exit\n\nThe spatial central-exit table reports a snapshot-position proxy only; it is not a claim about a No. 6 checking movement.\n\n## Training\n\nRehearse receiving pictures and passing corridors, while treating this as a static snapshot pilot rather than a complete tracking model.\n")
    decision=f"# StatsBomb 360 Sloan decision\n\nUsable 360 goalkeeper distributions: **{len(s)}** across **{s.match_id.nunique()}** matches. Coverage is **{coverage:.1f}%** of eligible goalkeeper distributions. Choice, central-exit, retention, and frozen-xR tables are delivered in `results/statsbomb360`. All results are associational and static-frame based.\n\n**MAIN PAPER: {'YES' if main_ok else 'NO'}**. This extension is classified **SUPPLEMENT / PROOF OF CONCEPT** unless all preregistered eligibility conditions are met. **ABSTRACT: NO**. Continuous player movement, off-camera options, and tactical viability remain unobserved.\n"
    (REP/'STATSBOMB360_SLOAN_DECISION.md').write_text(decision)
    print(f'R1.2 built: {len(s)} covered GK distributions, {len(opt)} visible candidates')
if __name__=='__main__': main()
