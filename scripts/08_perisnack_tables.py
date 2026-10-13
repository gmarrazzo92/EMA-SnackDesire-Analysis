#!/usr/bin/env python3
"""Regenerate the peri-snack statistics and figure data from public OSF data.

Matched controls are computed by 02_match_perisnack.R. Contrasts use pair-complete
ratings averaged within participants; participants are the statistical unit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--matched-pairs',type=Path, default=None)
a=p.parse_args()
root=a.project_root.resolve()
out=root/'results'/'08_perisnack_tables';out.mkdir(parents=True,exist_ok=True)
prompt=pd.read_csv(root/'data'/'derived'/'ema_prompt_reduced.csv',low_memory=False)
# The R matching script generates all four match sets, including the corrected
# pre-event sensitivity. This is the source used for all peri-snack tables.
match_file=a.matched_pairs or root/'results/02_match_perisnack/tables/15_matched_pairs_all.csv'
if not Path(match_file).is_file():
    raise SystemExit(f'Run 02_match_perisnack.R first; missing {match_file}')
matches=pd.read_csv(match_file,low_memory=False)
prompt['LowControl']=100-pd.to_numeric(prompt['Self_Control'],errors='coerce')
prompt['ID']=prompt['ID'].astype(str).str.replace(r'\.0$','',regex=True)
# Original row order is a data dependency: match indices refer to 1-based rows of
# the reconstructed prompt-level CSV, not to a sorted or reindexed copy.
prompt['timestamp']=pd.to_datetime(prompt.scheduled_datetime)
diff=prompt.groupby('ID',sort=False).timestamp.diff().dt.total_seconds()/3600
prompt['segment']=(diff.isna()|diff.le(0)|diff.gt(8)).groupby(prompt.ID).cumsum()
assert len(prompt)==16732, 'Input row count differs from audited sample'
VARS={'CD':'Snack desire intensity','LowControl':'Low perceived control','Stress':'Stress'}
cond_names={'snack':'Snack window','control':'Matched non-snack'}
primary='primary_time_only_no_reuse'
sensitive='sensitivity_time_plus_tminus1_CD_LowControl_no_reuse'
# Matching yield and quality: panel A of Supporting Information S1 Table.
matching_rows=[]
for match_set,m in matches.groupby('match_set',sort=False):
    matching_rows.append(dict(match_set=match_set,eligible_snack_events=2604,
        matched_events=len(m),participants=m.id.nunique(),
        unique_control_windows=m.control_row.nunique(),
        control_reuse_ratio=len(m)/m.control_row.nunique(),
        median_day_distance=m.day_distance.median(),
        median_hour_distance=m.hour_distance.median(),
        median_pre_CD_distance=m.pre_CD_distance.median() if m.pre_CD_distance.notna().any() else float('nan'),
        median_pre_LowControl_distance=m.pre_LowControl_distance.median() if m.pre_LowControl_distance.notna().any() else float('nan')))
pd.DataFrame(matching_rows).to_csv(out/'S1_Table_PanelA_matching_quality.csv',index=False)
summary=[]; tests=[]; pairdata=[]

def pick_values(m,side,offset,var):
    center=m[f'{side}_row'].to_numpy(dtype=int)-1
    ix=center+offset
    valid=(ix>=0)&(ix<len(prompt))
    result=np.full(len(ix),np.nan)
    if valid.any():
        safe=ix[valid]
        matching=(prompt.iloc[safe].segment.to_numpy()==prompt.iloc[center[valid]].segment.to_numpy()) & (prompt.iloc[safe].ID.to_numpy()==m.iloc[np.flatnonzero(valid)].id.astype(str).to_numpy())
        vals=pd.to_numeric(prompt.iloc[safe][var],errors='coerce').to_numpy(dtype=float)
        result[np.flatnonzero(valid)[matching]]=vals[matching]
    return result

def ttest(x):
    x=pd.Series(x,dtype=float).dropna().to_numpy()
    n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/np.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,se=se,t=mean/se,df=n-1,
          p=float(2*stats.t.sf(abs(mean/se),n-1)),ci_low=mean-stats.t.ppf(.975,n-1)*se,
          ci_high=mean+stats.t.ppf(.975,n-1)*se)

for match_set in [primary,sensitive]:
    m=matches[matches.match_set==match_set].copy()
    if match_set==primary: assert len(m)==2416,'Primary matching yield differs from manuscript'
    if match_set==sensitive: assert len(m)==2146,'Corrected sensitivity matching yield differs from manuscript'
    for var,name in VARS.items():
        raw={side:{off:pick_values(m,side,off,var) for off in (-2,-1,0,1,2)} for side in ['snack','control']}
        for off in (-2,-1,0,1,2):
            pair=pd.DataFrame({'id':m.id.to_numpy(),'snack':raw['snack'][off], 'control':raw['control'][off]}).dropna()
            if pair.empty: continue
            means=pair.groupby('id',sort=False)[['snack','control']].mean()
            for side in ['snack','control']:
                vals=means[side].to_numpy()
                sem=vals.std(ddof=1)/np.sqrt(len(vals));crit=stats.t.ppf(.975,len(vals)-1)
                summary.append(dict(match_set=match_set,variable=var,offset=off,condition=cond_names[side],n_participants=len(vals),n_pairs=len(pair),mean=float(vals.mean()),ci95_low=float(vals.mean()-sem*crit),ci95_high=float(vals.mean()+sem*crit)))
        # Two level contrasts, then the three shape contrasts; preserve pair completeness separately per contrast.
        measures={
            'level_tminus1':([-1],[1]),'level_tplus1':([1],[1]),
            'rise_m1_to_0':([-1,0],[-1,1]),
            'drop_0_to_p1':([0,1],[-1,1]),
            'peakness_0_vs_neighbors':([-1,0,1],[-.5,1,-.5]),
        }
        for metric,(offs,coefs) in measures.items():
            diffs=np.zeros(len(m))
            for off,coef in zip(offs,coefs):
                diffs+=coef*(raw['snack'][off]-raw['control'][off])
            person=pd.DataFrame({'id':m.id.to_numpy(),'difference':diffs}).dropna().groupby('id').difference.mean()
            res=ttest(person.to_numpy())
            tests.append(dict(match_set=match_set,variable=var,measure=metric,**res))

s=pd.DataFrame(summary);t=pd.DataFrame(tests)
t['family']=np.where(t.measure.str.startswith('level_'),'levels_6','shape_9')
t['q']=np.nan
for (mt,fam),ix in t.groupby(['match_set','family']).groups.items():
    t.loc[list(ix),'q']=multipletests(t.loc[list(ix),'p'],method='fdr_bh')[1]
s.to_csv(out/'participant_balanced_trajectories.csv',index=False)
t.to_csv(out/'participant_level_tests.csv',index=False)
# PLOS S1B and S2 source tables (with exact p and q in CSV).
t[(t.match_set==sensitive)&(t.family=='shape_9')].to_csv(out/'S1_Table_PanelB_matching_sensitivity.csv',index=False)
t[(t.match_set==primary)&(t.family=='levels_6')].to_csv(out/'S2_Table_PanelB_levels.csv',index=False)
t[(t.match_set==primary)&(t.family=='shape_9')].to_csv(out/'S2_Table_PanelC_shapes.csv',index=False)
s[(s.match_set==primary)&(s.offset.isin([-1,0,1]))].to_csv(out/'S2_Table_PanelA_means.csv',index=False)
# S1 Fig: reproduce the original plot from its source R code with equivalent
# facets, means and participant-level 95% t-based confidence bands.
plot=s[s.match_set==primary]; fig,axes=plt.subplots(1,3,figsize=(10.5,4.7),sharex=True,sharey=True)
for ax,(var,name) in zip(axes,VARS.items()):
    for side,col,ls,marker in [('control','#3D6A9A','--','o'),('snack','#BB4430','-','^')]:
        d=plot[(plot.variable==var)&(plot.condition==cond_names[side])].sort_values('offset')
        ax.fill_between(d.offset,d.ci95_low,d.ci95_high,color=col,alpha=.16,linewidth=0)
        ax.plot(d.offset,d['mean'],color=col,linestyle=ls,marker=marker,markersize=5,linewidth=2,label=cond_names[side])
    ax.axvline(0,color='#6B7078',linestyle=':',lw=.9)
    ax.set_title(name,fontsize=10.5,fontweight='bold')
    ax.set_xticks([-2,-1,0,1,2],['t−2','t−1','t0','t+1','t+2'])
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean EMA rating (0–100 VAS)')
fig.suptitle('Peri-snack patterns of reported desire intensity, low perceived control, and stress',fontweight='bold',fontsize=11.5,y=.99)
fig.text(.5,.88,'Participant-balanced means; bands are 95% CIs across participants',ha='center',fontsize=9,color='#6B7078')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.11,.89),ncol=2,frameon=False,fontsize=9)
fig.supxlabel('Prompt relative to focal window',y=.05)
fig.subplots_adjust(left=.08,right=.985,top=.73,bottom=.2,wspace=.12)
for ext in ['pdf','png','svg']:
    fig.savefig(out/f'S1_Fig_perisnack_trajectories.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
plt.close(fig)
# Sanity checks on exact manuscript values from *source data*, not hard-coded output.
for var,measure,want in [('CD','rise_m1_to_0',39.48),('CD','peakness_0_vs_neighbors',42.71),('LowControl','peakness_0_vs_neighbors',23.59),('Stress','peakness_0_vs_neighbors',3.02),('CD','level_tminus1',1.30),('CD','level_tplus1',-5.30)]:
    have=t.loc[(t.match_set==primary)&(t.variable==var)&(t.measure==measure),'mean'].iloc[0]
    if abs(have-want)>.015: raise AssertionError(f'Peri-snack validation failed for {var} {measure}: {have} vs {want}')
print('Reproduced S1 Fig and S1/S2 source tables with independently checked participant-level means and tests:',out)
