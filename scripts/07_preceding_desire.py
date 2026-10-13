#!/usr/bin/env python3
"""Test 23: t-1 binary desire, with NO snack at t-1; match original 15 pairs where available.

Always computes Test-A-aligned 2x2 (any subsequent snack). For original matched
primary, uses script-15 pair assignments, or recreates time/day greedy matching
from its source and labels the result as reconstructed until verified.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ROOT=Path(__file__).resolve().parents[1]

def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--testa-pairs',type=Path)
    ap.add_argument('--model-ready',type=Path)
    ap.add_argument('--matched-pairs',type=Path,help='Exact original 15_matched_pairs_all.csv; auto-detected when available')
    ap.add_argument('--output-dir',type=Path)
    return ap.parse_args()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def label_snack(s,amount):
    txt=s.astype('string').str.strip().str.lower()
    out=pd.Series(np.nan,index=s.index)
    out[txt.isin(['yes','y','true','1','snacked','si','sí'])]=1.
    out[txt.isin(['no','n','false','0'])]=0.
    other=pd.to_numeric(s,errors='coerce')
    miss=out.isna()&other.notna()
    out.loc[miss]=(other[miss]>0).astype(float)
    miss=out.isna()&amount.notna()
    out.loc[miss]=(amount[miss]>0).astype(float)
    return out

def prep_sequence(source):
    cols=['ID','Group','StudyDay','Day_nr','scheduled_datetime','CD','Snacked_YN','SA','LowControl','Self_Control']
    d=pd.read_csv(source,usecols=lambda c:c in cols,low_memory=False)
    d['row_original']=np.arange(1,len(d)+1,dtype=int)
    d['id']=d.ID.astype('string').str.replace(r'\.0$','',regex=True)
    d['group']=d.Group.astype('string').replace({'Medium':'Med'})
    d['timestamp']=pd.to_datetime(d.scheduled_datetime,errors='coerce')
    d['day']=pd.to_numeric(d['StudyDay'] if 'StudyDay' in d.columns else d['Day_nr'],errors='coerce')
    d['desire']=np.where(pd.to_numeric(d.CD,errors='coerce').notna(),
                         (pd.to_numeric(d.CD,errors='coerce')>0).astype(float),np.nan)
    amount=pd.to_numeric(d.SA,errors='coerce')
    d['snack']=label_snack(d.Snacked_YN,amount)
    d=d.loc[d.id.notna()&d.group.notna()&d.timestamp.notna()].copy()
    d=d.sort_values(['id','timestamp','row_original'],kind='stable').reset_index(drop=True)
    dd=d.groupby('id',sort=False)
    d['gap_prev']=(d.timestamp-dd.timestamp.shift(1)).dt.total_seconds()/3600
    d['new_seg']=d.gap_prev.isna()|d.gap_prev.le(0)|d.gap_prev.gt(8)
    d['segment_id']=d.groupby('id',sort=False).new_seg.cumsum()
    seg=d.groupby(['id','segment_id'],sort=False)
    d['prev_snack']=seg.snack.shift(1)
    d['prev_desire']=seg.desire.shift(1)
    d['prev_CD']=seg.CD.shift(1)
    d['next_snack']=seg.snack.shift(-1)
    d['has_core']=seg.cumcount().ge(1)&seg.cumcount(ascending=False).ge(1)
    d['hour']=d.timestamp.dt.hour+d.timestamp.dt.minute/60+d.timestamp.dt.second/3600
    return d

def recreate_matches(seq):
    """Exact core approach of script 15 primary: greedy, within-person, day/hour score, no reuse."""
    events=seq.loc[seq.has_core&seq.snack.eq(1), ['id','group','day','hour','row_original']].copy()
    ctrls=seq.loc[seq.has_core&seq.snack.eq(0), ['id','group','day','hour','row_original']].copy()
    events=events.sort_values(['id','day','hour','row_original'],kind='stable')
    chosen=[]
    for pid,evs in events.groupby('id',sort=False):
        c=ctrls.loc[ctrls.id==pid]
        used=np.zeros(len(c),dtype=bool)
        cvday=c.day.to_numpy(dtype=float); cvhour=c.hour.to_numpy(dtype=float)
        cvrow=c.row_original.to_numpy(dtype=int)
        for ev in evs.itertuples(index=False):
            if len(c)==0 or np.all(used): continue
            dh=np.abs(cvhour-ev.hour)
            dd=np.abs(cvday-ev.day)
            score=dh+0.10*dd
            valid=np.where(~used)[0]
            order=np.lexsort((cvrow[valid],dh[valid],dd[valid],score[valid]))
            k=valid[order[0]]
            used[k]=True
            chosen.append({'match_set':'primary_time_only_no_reuse', 'id':str(pid),
                           'snack_row':int(ev.row_original),'control_row':int(cvrow[k]),
                           'match_score':float(score[k])})
    return pd.DataFrame(chosen)

def stats_person(x):
    p=x.groupby('id',as_index=False).contrast.mean()
    vals=p.contrast.to_numpy(dtype=float)
    if len(vals)<3: raise ValueError('Not enough participant contrasts')
    res=stats.ttest_1samp(vals,0)
    sem=stats.sem(vals)
    ci=stats.t.interval(.95,df=len(vals)-1,loc=np.mean(vals),scale=sem)
    return dict(n_participants=len(p),mean_difference=float(np.mean(vals)),sem=float(sem),
                t=float(res.statistic),df=len(vals)-1,p_value=float(res.pvalue),
                CI_low=float(ci[0]),CI_high=float(ci[1]))

def main():
    a=parse_args(); root=a.project_root
    out=a.output_dir or root/'results/07_preceding_desire'
    tbl=out/'tables';tbl.mkdir(parents=True,exist_ok=True)
    testfile=a.testa_pairs or root/'results/03_gee/tables/21_testA_analysis_pairs_8h.csv'
    source=a.model_ready or root/'data/derived/ema_prompt_reduced.csv'
    old=a.matched_pairs or root/'results/02_match_perisnack/tables/15_matched_pairs_all.csv'
    for name,path in [('Test A pair file',testfile),('model-ready prompt file',source)]:
        if not path.exists(): raise FileNotFoundError(f'{name} missing: {path}')
    test=pd.read_csv(testfile,low_memory=False)
    tab=pd.crosstab(test.crave_t.astype(int),test.snack_next.astype(int))
    assert tab.shape==(2,2)
    or_raw=float(tab.loc[1,1]*tab.loc[0,0]/(tab.loc[1,0]*tab.loc[0,1]))
    descript={'comparison':'Test_A_eligible_unmatched', 'N_pairs':len(test),
             'n_participants':int(test.id.nunique()),'OR_raw':or_raw,
             'P_desire_before_snack':float(tab.loc[1,1]/tab[1].sum()),
             'P_desire_before_nonsnack':float(tab.loc[1,0]/tab[0].sum()),
             'P_snack_next_when_desire':float(tab.loc[1,1]/tab.loc[1].sum()),
             'P_snack_next_when_no_desire':float(tab.loc[0,1]/tab.loc[0].sum()),
             'n_00':int(tab.loc[0,0]),'n_01':int(tab.loc[0,1]),
             'n_10':int(tab.loc[1,0]),'n_11':int(tab.loc[1,1])}
    pd.DataFrame([descript]).to_csv(tbl/'23_testa_binary_descriptive.csv',index=False)
    seq=prep_sequence(source)
    if old.is_file():
        matches=pd.read_csv(old,low_memory=False)
        matches=matches[matches.match_set=='primary_time_only_no_reuse'].copy()
        mode='original_script15_saved_matches'
    else:
        matches=recreate_matches(seq)
        mode='reconstructed_script15_primary_matching_unverified_against_saved_pairs'
    if len(matches)==0: raise RuntimeError('No primary matches available')
    if matches.snack_row.duplicated().any() or matches.control_row.duplicated().any():
        raise ValueError('Primary matching contains reused focal or control rows')
    prev=seq.set_index('row_original')[['id','prev_snack','prev_desire','prev_CD','has_core','snack']]
    joined=matches[['id','snack_row','control_row']].copy()
    joined['id']=joined.id.astype('string').str.replace(r'\.0$','',regex=True)
    for side, col in [('snack','snack_row'),('control','control_row')]:
        d=prev.reindex(joined[col].to_numpy())
        joined[f'{side}_prev_snack']=d.prev_snack.to_numpy()
        joined[f'{side}_prev_desire']=d.prev_desire.to_numpy()
        joined[f'{side}_prev_CD']=pd.to_numeric(d.prev_CD, errors='coerce').to_numpy()
        joined[f'{side}_core']=d.has_core.to_numpy()
        joined[f'{side}_focal_snack']=d.snack.to_numpy()
        if not bool((joined['id'].to_numpy()==d.id.astype('string').to_numpy()).all()):
            raise ValueError(f'Original {side} rows do not correspond to matching participants; check inputs')
    # QA: original manuscript's unrestricted t-1 contrast was based on CD intensity.
    # Check whether reconstruction yields a similarly near-zero person-weighted contrast.
    original_like=joined.copy()
    original_like['contrast']=original_like.snack_prev_CD-original_like.control_prev_CD
    original_like=original_like.loc[original_like.contrast.notna()].copy()
    original_like_stats=stats_person(original_like)
    pd.DataFrame([{'source_mode':mode,'original_matched_pairs':len(joined),
        'complete_intensity_pairs':len(original_like),**original_like_stats}]).to_csv(
        tbl/'23_unrestricted_original_intensity_QA.csv',index=False)
    valid=joined.snack_prev_snack.eq(0)&joined.control_prev_snack.eq(0)&joined.snack_prev_desire.notna()&joined.control_prev_desire.notna()
    eligible=joined.loc[valid].copy()
    eligible['contrast']=eligible.snack_prev_desire-eligible.control_prev_desire
    eligible.to_csv(tbl/'23_primary_matched_restricted_pairs.csv',index=False)
    result={'source_mode':mode,'original_primary_matched_pairs':len(joined),
            'eligible_both_previous_nosnack_with_desire':len(eligible),
            'removed_by_restriction':int(len(joined)-len(eligible)),
            'P_desire_tminus1_before_snack':float(eligible.snack_prev_desire.mean()),
            'P_desire_tminus1_before_control':float(eligible.control_prev_desire.mean()),
            'pair_weighted_difference':float(eligible.contrast.mean()),
            **stats_person(eligible)}
    pd.DataFrame([result]).to_csv(tbl/'23_primary_matched_binary_tminus1.csv',index=False)
    manifest={'source_mode':mode,'testa_sha256':sha(testfile),'model_ready_sha256':sha(source),
              'original_match_sha256':sha(old) if old.is_file() else None,
              'Test_A':descript,'unrestricted_original_intensity_QA':original_like_stats,'matched':result}
    (out/'23_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Test A eligible risk-set: P(desire | next snack)=%.3f vs P(desire | no next snack)=%.3f, raw OR=%.3f' %
          (descript['P_desire_before_snack'],descript['P_desire_before_nonsnack'],or_raw))
    print('Matched pairs method:',mode)
    print('QA original unrestricted intensity at t-1: participant-mean difference %.3f, p=%.4g (%d pairs)'%(original_like_stats['mean_difference'],original_like_stats['p_value'],len(original_like)))
    print('Matched binary t-1, both prior non-snack: N=%d pairs, %d participants; desire %.3f vs %.3f; person-balanced diff=%.4f (95%% CI %.4f, %.4f), p=%.5g'%
          (len(eligible),result['n_participants'],result['P_desire_tminus1_before_snack'],result['P_desire_tminus1_before_control'],
           result['mean_difference'],result['CI_low'],result['CI_high'],result['p_value']))
    print('Saved:',out)

if __name__=='__main__': main()
