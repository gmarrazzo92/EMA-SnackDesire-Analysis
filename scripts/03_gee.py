#!/usr/bin/env python3
"""EMA Test A: Within-person current snack desire (with NO eating) and any next-window snacking.

Inputs and state reconstruction follow script 18_time_of_day_nominal_slot_surrogate_repro_v2_fixed_missingness.py.
Primary test: binary snack_{t+1}, ANY craving_{t+1}, from current snack_t=0.
Analysis: logistic GEE (population-average, cluster robust by ID, exchangeable working correlation),
with the exposure separated into within- and between-person components.
A subject-specific random-intercept GLMM using the same prepared pairs is in 04_glmm.R.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as e:
    raise SystemExit('Missing statsmodels. Run: python -m pip install -r requirements.txt') from e

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
INPUT_REL = Path('data/derived/ema_prompt_reduced.csv')
OUT_REL = Path('results/03_gee')


def args_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--input', type=Path, default=None, help='Override the input CSV')
    ap.add_argument('--output-dir', type=Path, default=None, help='Folder receiving tables and report')
    ap.add_argument('--gaps', default='8,4,2', help='Comma-separated maximum time gaps (hours)')
    ap.add_argument('--require-next-craving', action='store_true', help='Require next craving label as well, for exact four-state sample comparison')
    return ap.parse_args()


def binary_from_any(s: pd.Series) -> pd.Series:
    # Same positive/negative normalization as corrected script 18.
    txt = s.astype('string').str.strip().str.lower()
    yes = txt.isin(['yes', 'y', 'true', '1', 'snacked', 'si', 'sí'])
    no = txt.isin(['no', 'n', 'false', '0'])
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[yes] = 1.
    out.loc[no] = 0.
    unrecognized = s.notna() & out.isna()
    vals = pd.to_numeric(s.loc[unrecognized], errors='coerce')
    out.loc[vals.index[vals.notna()]] = (vals.loc[vals.notna()] > 0).astype(float)
    return out


def make_pairs(file: Path):
    required = ['ID', 'Group', 'scheduled_datetime', 'Prompt',
                'expected_prompts_per_day', 'StudyDay', 'CD', 'Snacked_YN', 'SA']
    d = pd.read_csv(file, usecols=lambda k: k in required + ['Craved_YN'], low_memory=False)
    absent = set(required) - set(d.columns)
    if absent:
        raise ValueError(f'Missing required fields in {file}: {sorted(absent)}')
    d['id'] = d['ID'].astype('string').str.replace(r'\.0$', '', regex=True)
    d['group'] = d['Group'].replace({'Medium': 'Med'}).astype('string')
    d['timestamp'] = pd.to_datetime(d['scheduled_datetime'], errors='coerce')
    d['day'] = pd.to_numeric(d['StudyDay'], errors='coerce')
    d['cd'] = pd.to_numeric(d['CD'], errors='coerce')
    d['crave_t'] = np.where(d['cd'].notna(), (d['cd'] > 0).astype(float), np.nan)
    d['snack_t'] = binary_from_any(d['Snacked_YN'])
    amounts = pd.to_numeric(d['SA'], errors='coerce')
    miss = d['snack_t'].isna() & amounts.notna()
    d.loc[miss, 'snack_t'] = (amounts.loc[miss] > 0).astype(float)
    d['prompt_number'] = pd.to_numeric(d['Prompt'], errors='coerce')
    d['expected'] = pd.to_numeric(d['expected_prompts_per_day'], errors='coerce')

    d = d.loc[d['id'].notna() & d['group'].notna() & d['timestamp'].notna()].copy()
    # Stable sort to match script 14/18 chronological adjacency.
    d = d.sort_values(['id', 'timestamp'], kind='stable').reset_index(drop=True)
    g = d.groupby('id', sort=False)
    d['gap_hours'] = (g['timestamp'].shift(-1) - d['timestamp']).dt.total_seconds() / 3600
    d['snack_next'] = g['snack_t'].shift(-1)
    d['crave_next'] = g['crave_t'].shift(-1)
    d['next_group'] = g['group'].shift(-1)
    d['next_prompt'] = g['prompt_number'].shift(-1)
    d['next_expected'] = g['expected'].shift(-1)
    d['next_slot'] = ((d['next_prompt'] - 1) % d['next_expected'] + 1)
    d['next_group_slot'] = d['next_group'].astype(str) + '_' + d['next_slot'].astype('Int64').astype(str)
    # Eligible link: same participant (guaranteed by groupwise shift), positive gap <=8h.
    tlink = d['gap_hours'].gt(0) & d['gap_hours'].le(8)
    state_t = d['crave_t'].notna() & d['snack_t'].notna()
    state_next = d['crave_next'].notna() & d['snack_next'].notna()
    audit = {
        'input_rows': len(d),
        'input_participants': int(d['id'].nunique()),
        'valid_temporal_links_8h': int(tlink.sum()),
        'four_state_labeled_rows': int(state_t.sum()),
        'complete_four_state_links_8h': int((tlink & state_t & state_next).sum()),
        'eligible_current_nonsnack_with_any_next_snack_8h': int((tlink & state_t & d['snack_t'].eq(0) & d['snack_next'].notna()).sum()),
    }
    # Diagnostic: never condition Test A on the next-window craving unless explicitly requested.
    d = d.loc[tlink & state_t & d['snack_t'].eq(0) & d['snack_next'].notna()].copy()
    d = d.loc[d['next_slot'].notna() & d['day'].notna() & d['gap_hours'].notna()]
    d = d.loc[d['next_expected'].isin([3, 6, 9])].copy()
    d['crave_t'] = d['crave_t'].astype(int)
    d['snack_next'] = d['snack_next'].astype(int)
    if 'Craved_YN' in d:
        yesno = binary_from_any(d['Craved_YN'])
        audit['current_craved_yesno_disagreement'] = int((yesno.notna() & yesno.ne(d['crave_t'])).sum())
    return d, audit


def make_model_data(all_pairs: pd.DataFrame, max_gap: float) -> pd.DataFrame:
    d = all_pairs.loc[all_pairs['gap_hours'] <= max_gap].copy()
    # Explicit within-between (Mundlak) decomposition computed on each analysis risk set.
    d['crave_person_mean'] = d.groupby('id')['crave_t'].transform('mean')
    d['crave_within'] = d['crave_t'] - d['crave_person_mean']
    d['study_day_c'] = d['day'] - d['day'].mean()
    d['log_gap'] = np.log(d['gap_hours'])
    return d


def model_gee(d: pd.DataFrame):
    # No separate Group term: next_group_slot includes the monitoring group and scheduled time-of-day.
    spec = ('snack_next ~ crave_within + crave_person_mean + '
            'C(next_group_slot) + study_day_c + log_gap')
    family = sm.families.Binomial()
    m = smf.gee(spec, groups='id', data=d, family=family,
                cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
    if not np.all(np.isfinite(m.params)):
        raise RuntimeError('Non-finite model parameters; inspect group-slot cell counts')
    return m


def risk_summary(d: pd.DataFrame, gap: float):
    rows = []
    for name, dd in [('overall', d)] + [(str(v), g) for v, g in d.groupby('group', observed=True)]:
        for exposure in [0, 1]:
            x = dd.loc[dd['crave_t'] == exposure]
            rows.append({'max_gap_h': gap, 'group': name, 'craving_t': exposure,
                         'pairs': int(len(x)), 'participants': int(x['id'].nunique()),
                         'snacks_next': int(x['snack_next'].sum()),
                         'next_snack_probability': float(x['snack_next'].mean()) if len(x) else np.nan})
    return rows


def model_row(m, gap: float, d: pd.DataFrame) -> dict:
    coef = float(m.params['crave_within'])
    se = float(m.bse['crave_within'])
    # Robust sandwich confidence intervals from GEE.
    low, high = m.conf_int().loc['crave_within'].tolist()
    p = float(m.pvalues['crave_within'])
    # Standardize current desire to 0 or 1 at each observed person, slot and gap.
    d0 = d.copy()
    d1 = d.copy()
    d0['crave_within'] = -d0['crave_person_mean']
    d1['crave_within'] = 1 - d1['crave_person_mean']
    prob0 = np.asarray(m.predict(d0), dtype=float)
    prob1 = np.asarray(m.predict(d1), dtype=float)
    # Person-balanced standardized probabilities, so heavy responders do not dominate summaries.
    predictions = pd.DataFrame({'id': d['id'].values, 'p0': prob0, 'p1': prob1})
    pmeans = predictions.groupby('id')[['p0', 'p1']].mean()
    return {'max_gap_h': gap, 'n_pairs': len(d), 'n_participants': d['id'].nunique(),
            'n_participants_with_both_exposures': int(d.groupby('id')['crave_t'].nunique().eq(2).sum()),
            'n_snacks_next': int(d['snack_next'].sum()),
            'method': 'GEE_binomial_logit_exchangeable_robust',
            'beta_within': coef, 'robust_se': se, 'odds_ratio': float(np.exp(coef)),
            'OR_CI_low': float(np.exp(low)), 'OR_CI_high': float(np.exp(high)),
            'p_value': p, 'converged': bool(m.converged),
            'std_probability_c0': float(pmeans['p0'].mean()),
            'std_probability_c1': float(pmeans['p1'].mean()),
            'std_risk_difference': float((pmeans['p1'] - pmeans['p0']).mean())}


def main():
    args = args_parser()
    gaps = sorted({float(x) for x in args.gaps.split(',') if x.strip()}, reverse=True)
    if not gaps or any(g <= 0 or g > 8 for g in gaps):
        raise ValueError('--gaps must be between 0 and 8 hours (e.g. 8,4,2)')
    infile = args.input or (args.project_root / INPUT_REL)
    outdir = args.output_dir or (args.project_root / OUT_REL)
    if not infile.is_file():
        raise FileNotFoundError(f'Missing original input CSV: {infile}. Pass --input FILE if located elsewhere.')
    tables = outdir / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    base, audit = make_pairs(infile)
    if args.require_next_craving:
        base = base.loc[base['crave_next'].notna()].copy()
    audit['require_next_craving'] = args.require_next_craving
    audit['complete_case_risk_set_after_option'] = len(base)
    audit['eligible_unique_participants'] = int(base['id'].nunique())
    # Cross-check standard manuscript sample. Informative; not a hard restriction.
    if audit['complete_four_state_links_8h'] != 10980:
        warnings.warn('Four-state links differ from manuscript (expected 10,980). Verify version and input.')

    rows = []
    desc = []
    model_text = []
    all_coefficients = []
    for gap in gaps:
        d = make_model_data(base, gap)
        if len(d) < 100 or d['crave_t'].nunique() < 2:
            raise RuntimeError(f'Insufficient observations for gap {gap}: {len(d)}')
        desc += risk_summary(d, gap)
        m = model_gee(d)
        r = model_row(m, gap, d)
        ci_all=m.conf_int()
        for term in m.params.index:
            beta=float(m.params[term]); low=float(ci_all.loc[term,0]); high=float(ci_all.loc[term,1])
            all_coefficients.append(dict(max_gap_h=gap, model='GEE', term=term,
                beta=beta, robust_se=float(m.bse[term]), z=float(beta/m.bse[term]),
                p_value=float(m.pvalues[term]), beta_ci_low=low, beta_ci_high=high,
                odds_ratio=float(np.exp(beta)), OR_ci_low=float(np.exp(low)),OR_ci_high=float(np.exp(high)),
                n_pairs=len(d),n_participants=d.id.nunique()))
        rows.append(r)
        model_text += [f'=== Maximum gap {gap:g} hours ===',
                       f'Pairs: {len(d)}, participants: {d.id.nunique()}',
                       str(m.summary()), '',
                       'NOTE: GEE OR is population-average, not a random-intercept subject-specific OR.',
                       'Unresolved (craving/no snack) vs no-craving/no-snack within person;',
                       'outcome any snacking at t+1, with no restriction on craving at t+1.', '']
        print(f"Gap<={gap:g}h: N={r['n_pairs']}, ID={r['n_participants']}, "
              f"OR={r['odds_ratio']:.3f} ({r['OR_CI_low']:.3f}-{r['OR_CI_high']:.3f}), "
              f"p={r['p_value']:.4g}, converged={r['converged']}")
        if gap == 8:
            d.to_csv(tables / '21_testA_analysis_pairs_8h.csv', index=False)
    pd.DataFrame(all_coefficients).to_csv(tables / '21_testA_gee_full_coefficients.csv',index=False)
    pd.DataFrame([audit]).to_csv(tables / '21_testA_flow_audit.csv', index=False)
    pd.DataFrame(desc).to_csv(tables / '21_testA_descriptives_by_gap_group.csv', index=False)
    pd.DataFrame(rows).to_csv(tables / '21_testA_gee_results_by_gap.csv', index=False)
    (outdir / '21_testA_gee_model_details.txt').write_text('\n'.join(model_text), encoding='utf-8')
    (outdir / '21_testA_run_manifest.json').write_text(json.dumps({
        'input': str(infile.resolve()), 'output': str(outdir.resolve()),
        'max_gaps_h': gaps, 'require_next_craving': args.require_next_craving,
        'primary_8h_gaps': 8 in gaps, 'software': {'statsmodels': sm.__version__, 'pandas': pd.__version__},
        'operationalization': 'CD>0 at t (no snack); any snack at t+1; same participant; 0<gap<=8',
        'note': 'Exploratory association, not a causal estimate.'
    }, indent=2), encoding='utf-8')
    print('Results written to:', outdir)
    print('Main output:', tables / '21_testA_gee_results_by_gap.csv')
    print('For subject-specific GLMM run scripts/04_glmm.R on the 8h analysis-pairs CSV.')

if __name__ == '__main__':
    main()
