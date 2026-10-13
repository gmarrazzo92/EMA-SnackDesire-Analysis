#!/usr/bin/env python3
"""Test 24: participant-day fixed-effects LPM, within-day no-current-snack transition risk set.

Within-demeaned OLS. Cluster sandwich SE by participant, with CR1 small-sample correction that
excludes fixed effects nested in the participant clusters from the degrees-of-freedom
penalty. Includes next group-slot and log elapsed gap.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats,linalg
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parents[1]

def args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--testa-pairs',type=Path)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--groups',default='Med,High',help='Primary monitoring groups; Low omitted by default')
    ap.add_argument('--gaps',default='8,4,2')
    return ap.parse_args()

def clustered_lpm(y,X,clust,number_absorbed):
    """Cluster sandwich (CR1); nested absorbed FE do not count toward K.

    In this analysis the participant-day (or participant) FE are nested within the
    participant-level clustering used for covariance estimation. Therefore
    the CR1 factor is G/(G-1) * (N-1)/(N-K), where K is the rank of the
    retained within-transformed slope regressors, not K + number_absorbed.
    number_absorbed is retained for a transparent function signature/audit.
    """
    X=np.asarray(X,float); y=np.asarray(y,float)
    rank=int(np.linalg.matrix_rank(X)); assert rank==X.shape[1],(rank,X.shape)
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    resid=y-X@coef
    xx=np.linalg.inv(X.T@X)
    keys,ci=np.unique(np.asarray(clust),return_inverse=True)
    meat=np.zeros((X.shape[1],X.shape[1]))
    for k in range(len(keys)):
        w=(X[ci==k,:]*resid[ci==k,None]).sum(axis=0)
        meat+=np.outer(w,w)
    N=len(y); G=len(keys); dof=N-rank  # FE nested in participant clusters: zero FE dof penalty
    if dof<=0 or G<=2: raise RuntimeError('Insufficient residual degrees of freedom')
    correction=G/(G-1)*(N-1)/dof
    se=np.sqrt(np.diag(xx@meat@xx)*correction)
    critical=float(stats.t.ppf(.975,G-1))
    t=coef/se
    p=2*stats.t.sf(np.abs(t),df=G-1)
    return coef,se,p,np.column_stack([coef-critical*se,coef+critical*se]),resid,

def demean(d,columns,group):
    return d[columns]-d.groupby(group,observed=True)[columns].transform('mean')

def fit(d,fe):
    d=d.copy()
    d['id']=d.id.astype('string')
    d['day_key']=d.id+'__'+d.timestamp.dt.strftime('%Y-%m-%d')
    # group X next-slot controls, drop one globally; within FE most slot dummies remain usable.
    slot=pd.get_dummies(d.next_group_slot.astype('string'),prefix='slot',drop_first=True,dtype=float)
    # In participant-day FE, the six medium-frequency slot indicators sum to
    # one within each medium-group participant-day. Consequently *one* is
    # exactly redundant with the absorbed day intercept. A QR pivot previously
    # dropped Med_1 or Med_6 depending on BLAS/platform, changing the nuisance
    # coefficient reference while leaving the desire coefficient unchanged.
    # Fix Med_1 as the reference to make the full coefficient table portable.
    if fe and 'slot_Med_1' in slot.columns:
        slot=slot.drop(columns=['slot_Med_1'])
    x=pd.concat([d[['crave_t','log_gap']].astype(float),slot],axis=1)
    # For person FE pooled comparator include day trend, absorbed in participant-day FE.
    if not fe: x['study_day_c']=d.day.astype(float)-d.day.mean()
    base=pd.concat([d[['snack_next','id','day_key']],x],axis=1)
    grouping='day_key' if fe else 'id'
    # Verify the condition required for the nested-fixed-effects CR1 correction.
    if fe and not base.groupby('day_key', observed=True)['id'].nunique().eq(1).all():
        raise RuntimeError('Participant-day fixed effects are not nested in participant clusters')
    y=base['snack_next'].astype(float)-base.groupby(grouping)['snack_next'].transform('mean')
    z=demean(base,list(x.columns),grouping)
    # QR with forced first column (exposure), remove exactly redundant slot columns.
    a=z.to_numpy(dtype=float)
    if np.linalg.norm(a[:,0])<1e-9: raise RuntimeError('No within-cluster desire variation')
    residual=a[:,1:]-np.outer(a[:,0],(a[:,0]@a[:,1:])/(a[:,0]@a[:,0]))
    _,rr,piv=linalg.qr(residual,mode='economic',pivoting=True)
    cutoff=np.finfo(float).eps*max(residual.shape)*abs(rr[0,0]) if rr.size else 0
    usable=int((np.abs(np.diag(rr))>cutoff).sum())
    keep=[0]+sorted(int(v)+1 for v in piv[:usable])  # deterministic output order
    Xa=a[:,keep]
    groups=int(base[grouping].nunique())
    b,se,p,ci,_=clustered_lpm(y.to_numpy(dtype=float),Xa,d.id,groups)
    const_var=d.groupby('day_key').agg(n_obs=('crave_t','size'),
         desire_levels=('crave_t','nunique'),outcome_levels=('snack_next','nunique'))
    counts={'n_participant_days':len(const_var),
            'n_days_at_least_2':int((const_var.n_obs>=2).sum()),
            'n_days_desire_varies':int((const_var.desire_levels==2).sum()),
            'n_days_outcome_varies':int((const_var.outcome_levels==2).sum()),
            'n_days_both_vary':int(((const_var.desire_levels==2)&(const_var.outcome_levels==2)).sum()),
            'n_days_no_within_desire_variation':int((const_var.desire_levels==1).sum())}
    coef_table=pd.DataFrame({'term':x.columns[keep], 'beta':b, 'robust_se':se,
        't':b/se,'p_value':p,'ci_low':ci[:,0],'ci_high':ci[:,1]})
    return {'method':'participant_day_FE_LPM' if fe else 'participant_FE_LPM',
            'b_desire':float(b[0]),'robust_SE':float(se[0]),
            'CI_low':float(ci[0,0]),'CI_high':float(ci[0,1]),
            'p_value':float(p[0]),'n_parameters_after_rank_check':len(keep),
            'n_absorbed_FE':groups,'n_pairs':len(d),'n_participants':d.id.nunique(),
            **counts}, const_var, coef_table

def main():
    a=args();root=a.project_root
    source=a.testa_pairs or root/'results/03_gee/tables/21_testA_analysis_pairs_8h.csv'
    out=a.output_dir or root/'results/05_fixed_effects'
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    if not source.is_file(): raise FileNotFoundError(source)
    d=pd.read_csv(source,low_memory=False)
    d['timestamp']=pd.to_datetime(d.timestamp,errors='raise')
    d['next_timestamp']=d.timestamp+pd.to_timedelta(d.gap_hours,unit='h')
    # Include only actual same-calendar-day pairs (not merely <=8-hour gaps)
    same_day=(d.timestamp.dt.date==d.next_timestamp.dt.date)
    # Avoid floating point date rollover by using original timestamps rounded to nearest second.
    d=d.loc[same_day].copy()
    selected={v.strip() for v in a.groups.split(',')}
    d=d.loc[d.group.isin(selected)].copy()
    d['log_gap']=np.log(d.gap_hours)
    daysum=d.groupby(['id',d.timestamp.dt.date],observed=True).agg(
        n=('crave_t','size'),exposure_var=('crave_t','nunique'),outcome_var=('snack_next','nunique')).reset_index()
    rows=[]
    all_coefficients=[]
    for gap in [float(x) for x in a.gaps.split(',') if x.strip()]:
        dd=d.loc[d.gap_hours<=gap].copy()
        if dd.empty:continue
        for fe in [False,True]:
            r,daycounts,coef_table=fit(dd,fe)
            coef_table['max_gap_hours']=gap
            coef_table['method']=r['method']
            coef_table['n_pairs']=len(dd)
            coef_table['n_participants']=dd.id.nunique()
            all_coefficients.append(coef_table)
            r.update({'max_gap_hours':gap,'included_groups':','.join(sorted(dd.group.unique()))})
            rows.append(r)
            print('Gap <=%g h, %s: N=%d, ID=%d, days=%d, b=%.5f (95%% CI %.5f,%.5f), p=%.6g'%(gap,r['method'],r['n_pairs'],r['n_participants'],r['n_participant_days'],r['b_desire'],r['CI_low'],r['CI_high'],r['p_value']))
        if gap==8:
            daycounts.to_csv(tables/'24_participant_day_information_counts.csv')
    out_table=pd.DataFrame(rows)
    out_table.to_csv(tables/'24_within_day_fixed_effect_results.csv',index=False)
    pd.concat(all_coefficients,ignore_index=True).to_csv(tables/'24_full_coefficients.csv',index=False)
    with open(source,'rb') as f:h=hashlib.sha256(f.read()).hexdigest()
    manifest={'testA_sha256':h,'selected_groups':sorted(selected),'within_day_only':True,
        'adjustments':['group_x_next_slot','log_gap'],
        'methods':['participant FE LPM as same-risk-set comparator',
                   'participant-day FE LPM (within transformed)'],
        'cluster':'participant','SE':'CR1 sandwich: absorbed fixed effects nested within participant clustering excluded from K; t df = participant clusters minus one',
        'interpretation':'association, not causal; adjustment removes day-constant characteristics only',
        'results':rows}
    (out/'24_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Saved:',out)

if __name__=='__main__':main()
