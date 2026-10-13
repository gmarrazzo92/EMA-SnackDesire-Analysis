#!/usr/bin/env python3
"""Constrained 5,000-permutation surrogate from the manuscript.

Preserves the observed desire series, snack report missingness, and the
participant x prompt-slot x same-window-desire margins. Does NOT preserve
snack autocorrelation or all day-level circumstances (see manuscript).
Extracted without the unused diagnostic B1/B2 regression models.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
INPUT=Path('data/derived/ema_prompt_reduced.csv')
OUT=Path('results/06_constrained_surrogate')

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--input',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--permutation-gaps',default='8,2')
    p.add_argument('--B',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261009)
    return p.parse_args()

def yes_no(series):
    t = series.astype('string').str.strip().str.lower()
    y = t.isin(['yes', 'y', 'true', '1', 'snacked', 'si', 'sí'])
    n = t.isin(['no', 'n', 'false', '0'])
    x = pd.Series(np.nan, index=series.index, dtype=float)
    x.loc[y] = 1.
    x.loc[n] = 0.
    remaining = series.notna() & x.isna()
    parsed = pd.to_numeric(series.loc[remaining], errors='coerce')
    x.loc[parsed.index[parsed.notna()]] = (parsed.loc[parsed.notna()] > 0).astype(float)
    return x


def prepare(path):
    cols = ['ID','Group','scheduled_datetime','Prompt','expected_prompts_per_day',
            'StudyDay','CD','Snacked_YN','SA']
    d = pd.read_csv(path, usecols=lambda k: k in cols + ['Craved_YN'], low_memory=False)
    missing = set(cols) - set(d.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    d['id'] = d['ID'].astype('string').str.replace(r'\.0$', '', regex=True)
    d['group'] = d['Group'].replace({'Medium':'Med'}).astype('string')
    d['time'] = pd.to_datetime(d['scheduled_datetime'], errors='coerce')
    d['day'] = pd.to_numeric(d['StudyDay'], errors='coerce')
    cd = pd.to_numeric(d['CD'], errors='coerce')
    d['crave'] = np.where(cd.notna(), (cd > 0).astype(float), np.nan)
    d['snack'] = yes_no(d['Snacked_YN'])
    sa = pd.to_numeric(d['SA'], errors='coerce')
    fill = d['snack'].isna() & sa.notna()
    d.loc[fill,'snack'] = (sa.loc[fill] > 0).astype(float)
    prompt = pd.to_numeric(d['Prompt'], errors='coerce')
    expected = pd.to_numeric(d['expected_prompts_per_day'], errors='coerce')
    d['slot'] = ((prompt - 1) % expected + 1)
    d['expected'] = expected
    d = d.loc[d['id'].notna() & d['group'].notna() & d['time'].notna()].copy()
    d = d.sort_values(['id','time'], kind='stable').reset_index(drop=True)
    grp = d.groupby('id', sort=False)
    d['gap_hours'] = (grp['time'].shift(-1) - d['time']).dt.total_seconds()/3600
    d['crave_next'] = grp['crave'].shift(-1)
    d['snack_next'] = grp['snack'].shift(-1)
    d['next_slot'] = grp['slot'].shift(-1)
    d['next_group'] = grp['group'].shift(-1)
    d['next_expected'] = grp['expected'].shift(-1)
    d['next_group_slot'] = d['next_group'].astype(str)+'_'+d['next_slot'].astype('Int64').astype(str)
    d['valid8'] = d['gap_hours'].gt(0) & d['gap_hours'].le(8)
    labeled = d['crave'].notna() & d['snack'].notna()
    full_link = d['valid8'] & labeled & d['crave_next'].notna() & d['snack_next'].notna()
    risk = d['valid8'] & labeled & d['snack'].eq(0) & d['snack_next'].notna()
    risk &= d['day'].notna() & d['next_slot'].notna() & d['next_expected'].isin([3,6,9])
    audit = {'input_rows':int(len(d)), 'participant_count':int(d['id'].nunique()),
             'state_labeled_rows':int(labeled.sum()),
             'complete_four_state_links_8h':int(full_link.sum()),
             'testA_reconstructed_pairs_8h':int(risk.sum()),
             'testB_complete_next_desire_pairs_8h':int((risk & d['crave_next'].notna()).sum()),
             'next_desire_missing_from_testA_pairs':int((risk & d['crave_next'].isna()).sum())}
    if audit['state_labeled_rows']!=14689 or audit['complete_four_state_links_8h']!=10980 or audit['testA_reconstructed_pairs_8h']!=8492:
        raise RuntimeError(f'Input does not reconstruct v3 manuscript / Test A: {audit}')
    return d,audit


def make_shuffler(d):
    # Preserve C at every assessment (including missingness) and shuffle *only*
    # observed S across days with matched ID x nominal slot x simultaneous C.
    c = d['crave'].to_numpy(dtype=float)
    s = d['snack'].to_numpy(dtype=float)
    valid = np.isfinite(c) & np.isfinite(s) & d['slot'].notna().to_numpy()
    idx_by_group = {}
    for key,loc in d.loc[valid].groupby(['id','slot','crave'],sort=False).indices.items():
        # groupby.indices on subset yields positions in subset, not in original d
        idx_by_group[key]=np.flatnonzero(valid)[np.asarray(loc,dtype=np.int64)]
    all_groups=[v for v in idx_by_group.values() if len(v)>=2]
    varying_groups=[v for v in all_groups if np.unique(s[v]).size==2]
    return c,s,varying_groups,{'shuffle_strata_total':len(idx_by_group),
                                'shuffle_strata_size_ge2':len(all_groups),
                                'shuffle_strata_with_both_snack_values':len(varying_groups),
                                'nonmissing_joint_state_rows':int(valid.sum())}


def stat_fn(c,s,links,id_codes,n_ids):
    # links are chronological pairs satisfying the gap rule; the risk set is
    # recalculated for each randomized snack series, because S_t may change.
    ni=links+1
    ct=c[links]; st=s[links]; sn=s[ni]; cn=c[ni]
    valid = np.isfinite(ct)&np.isfinite(st)&np.isfinite(sn)&(st==0)
    risk1=valid&(ct==1)
    risk0=valid&(ct==0)
    n1=int(risk1.sum()); n0=int(risk0.sum())
    if not n1 or not n0:
        return None
    y1=float(sn[risk1].sum()); y0=float(sn[risk0].sum())
    p1=y1/n1; p0=y0/n0
    rd=p1-p0
    # OR corrected only when an extreme zero-cell is encountered.
    odds=lambda y,n: (y+.5)/(n-y+.5)
    crude_or=odds(y1,n1)/odds(y0,n0)
    edge_denom=risk1&np.isfinite(cn)
    edge_num=edge_denom&(cn==1)&(sn==1)
    edge_prob=float(edge_num.sum()/edge_denom.sum()) if edge_denom.any() else np.nan
    # Equal-weight within-person difference, restricted to persons with both
    # exposure states at that particular permutation. It is descriptive, not GEE.
    ii=id_codes[links]
    a=np.bincount(ii[risk1],minlength=n_ids)
    b=np.bincount(ii[risk0],minlength=n_ids)
    ay=np.bincount(ii[risk1],weights=sn[risk1],minlength=n_ids)
    by=np.bincount(ii[risk0],weights=sn[risk0],minlength=n_ids)
    both=(a>0)&(b>0)
    balanced_rd=float(np.mean(ay[both]/a[both]-by[both]/b[both])) if both.any() else np.nan
    return {'testA_raw_RD':rd,'testA_raw_OR':crude_or,
            'testA_participant_balanced_RD':balanced_rd,
            'original_edge_C1S0_to_C1S1':edge_prob,
            'risk_n_c1':n1,'risk_n_c0':n0,
            'edge_count':int(edge_num.sum()),'edge_denom':int(edge_denom.sum()),
            'ids_with_both_states':int(both.sum())}


def perm_test(d,gap,B,seed):
    c,s,groups,shuffle_audit=make_shuffler(d)
    links=np.flatnonzero((d['gap_hours'].to_numpy(dtype=float)>0)&
                         (d['gap_hours'].to_numpy(dtype=float)<=gap))
    # Link indexing safe because within-ID shift defines gap; cross-ID gap is NaN.
    ids,nids=pd.factorize(d['id'],sort=False)
    obs=stat_fn(c,s,links,ids,len(nids))
    if obs is None: raise RuntimeError('No valid observed test statistic')
    if gap==8 and not np.isclose(obs['original_edge_C1S0_to_C1S1'],.3419,atol=.0001):
        raise RuntimeError(f'Original four-state edge does not match manuscript: {obs}')
    rng=np.random.default_rng(seed)
    targets=['testA_raw_RD','testA_raw_OR','testA_participant_balanced_RD','original_edge_C1S0_to_C1S1']
    sims=[]
    for b in range(B):
        randomized=s.copy()
        for ix in groups:
            randomized[ix]=rng.permutation(randomized[ix])
        if b==0:
            if not np.array_equal(np.isnan(s),np.isnan(randomized)):
                raise RuntimeError('Permutation moved missing snack labels')
            # Verify preserved within-slot same-window association counts.
            for ix in groups:
                if randomized[ix].sum()!=s[ix].sum():
                    raise RuntimeError('Permutation did not preserve within-stratum snack counts')
        stats=stat_fn(c,randomized,links,ids,len(nids))
        if stats is None: raise RuntimeError('Empty permuted risk set')
        sims.append({'permutation':b+1,'gap_h':gap,**stats})
        if (b+1)%1000==0 or b+1==B:
            print(f'  coupling-preserving shuffle gap <= {gap:g} h: {b+1}/{B}',flush=True)
    sims=pd.DataFrame(sims)
    summary=[]
    for t in targets:
        vals=sims[t].to_numpy(dtype=float)
        observed=obs[t]
        upper=(1+np.sum(vals>=observed))/(B+1)
        lower=(1+np.sum(vals<=observed))/(B+1)
        summary.append({'gap_h':gap,'metric':t,'observed':observed,
                        'null_mean':float(vals.mean()),'null_sd':float(vals.std(ddof=1)),
                        'null_p025':float(np.quantile(vals,.025)),
                        'null_p975':float(np.quantile(vals,.975)),
                        'observed_minus_null_mean':float(observed-vals.mean()),
                        'p_upper':upper,'p_lower':lower,
                        'p_two_sided':min(1.,2*min(upper,lower)),
                        'B':B,'seed':seed})
    return pd.DataFrame(summary),sims,obs,shuffle_audit


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def parse_gaps(s):
    result=sorted({float(z.strip()) for z in s.split(',') if z.strip()},reverse=True)
    if not result or any(x<=0 or x>8 for x in result):
        raise ValueError('Gaps must fall in (0,8] hours')
    return result


def main():
    args=cli()
    if args.B<1:raise ValueError('B must be positive')
    pgaps=parse_gaps(args.permutation_gaps)
    inp=args.input or args.project_root/INPUT
    out=Path(args.output_dir or args.project_root/OUT)
    if not inp.exists():raise FileNotFoundError(inp)
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    d,audit=prepare(inp)
    summaries=[];sims_all=[];perm_audits={}
    for gap in pgaps:
        s,sim,ob,au=perm_test(d,gap,args.B,args.seed)
        summaries.append(s);sims_all.append(sim)
        perm_audits[str(gap)]={'observed':ob,**au}
        for _,row in s.iterrows():
            print(f'Gap <= {gap:g} h {row.metric}: observed={row.observed:.6f}, null mean={row.null_mean:.6f}, p={row.p_upper:.6g}',flush=True)
    pd.concat(summaries,ignore_index=True).to_csv(tables/'22_testB_conditional_shuffle_summary.csv',index=False)
    pd.concat(sims_all,ignore_index=True).to_csv(tables/'22_testB_conditional_shuffle_iterations.csv',index=False)
    (out/'surrogate_manifest.json').write_text(json.dumps({'source':'Constrained surrogate, extracted from original 22 script',
        'input_sha256':sha256_file(inp),'B':args.B,'seed':args.seed,
        'gaps':pgaps,'source_reconstruction':audit,'permutation_audit':perm_audits},indent=2),encoding='utf-8')
    print('Saved surrogate results to',out)
if __name__=='__main__':main()
