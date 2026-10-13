#!/usr/bin/env python3
"""Create compact, manuscript-facing result CSVs from freshly computed outputs.
Full regression source CSVs are created by scripts 03, 04 and 05.
No unpublished exploratory sequence/Markov outputs are generated.
"""
from pathlib import Path
import shutil, argparse
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
r=a.project_root.resolve();o=r/'results'/'09_build_tables';o.mkdir(parents=True,exist_ok=True)
A=pd.read_csv(r/'results/03_gee/tables/21_testA_gee_results_by_gap.csv')
FE=pd.read_csv(r/'results/05_fixed_effects/tables/24_within_day_fixed_effect_results.csv')
T=pd.read_csv(r/'results/07_preceding_desire/tables/23_primary_matched_binary_tminus1.csv')
SUR=pd.read_csv(r/'results/06_constrained_surrogate/tables/22_testB_conditional_shuffle_summary.csv')
rows=[]
for _,z in A.sort_values('max_gap_h',ascending=False).iterrows():
 rows.append(dict(analysis=f'GEE (≤{z.max_gap_h:g} h)',participants=int(z.n_participants),pairs=int(z.n_pairs),effect_type='OR',estimate=z.odds_ratio,ci_low=z.OR_CI_low,ci_high=z.OR_CI_high,p_value=z.p_value))
for _,z in FE.iterrows():
 if (z.max_gap_hours==4): continue # same same-day pairs as 8h; not a manuscript row
 if z.max_gap_hours==2 and z.method=='participant_FE_LPM':continue
 rows.append(dict(analysis=f'{z.method} (≤{z.max_gap_hours:g} h)',participants=int(z.n_participants),pairs=int(z.n_pairs),effect_type='probability difference',estimate=z.b_desire,ci_low=z.CI_low,ci_high=z.CI_high,p_value=z.p_value))
z=SUR[(SUR.gap_h==8)&(SUR.metric=='testA_raw_RD')].iloc[0]
rows.append(dict(analysis='Constrained surrogate (≤8 h)',participants=136,pairs=8492,
    effect_type='unadjusted probability difference, surrogate mean in note',estimate=z.observed,
    ci_low=float('nan'),ci_high=float('nan'),p_value=z.p_upper,
    surrogate_mean=z.null_mean,n_surrogates=z.B))
glmm_path=r/'results/04_glmm/tables/21_testA_glmm_full_fixed_effects.csv'
if glmm_path.is_file():
 d=pd.read_csv(glmm_path);z=d.loc[d.term=='crave_within'].iloc[0]
 rows.append(dict(analysis='Random-intercept GLMM (≤8 h)',participants=int(z.n_participants),pairs=int(z.n_pairs),effect_type='OR',estimate=z.odds_ratio,ci_low=z.OR_ci_low,ci_high=z.OR_ci_high,p_value=z.p_value))
else: print('GLMM not generated: Rscript/lme4 unavailable, manuscript Table 2 is incomplete until R run.')
# Matched difference is a proportion, multiply by 100 only when preparing manuscript display.
z=T.iloc[0]
rows.append(dict(analysis='Matched preceding-window desire',participants=int(z.n_participants),pairs=int(z.eligible_both_previous_nosnack_with_desire),effect_type='probability difference',estimate=z.mean_difference,ci_low=z.CI_low,ci_high=z.CI_high,p_value=z.p_value))
pd.DataFrame(rows).to_csv(o/'Table2_model_estimates.csv',index=False)
# Do not label adjusted and unadjusted probabilities as the same estimand.
# Full coefficients exported directly from their analysis scripts.
g=pd.read_csv(r/'results/03_gee/tables/21_testA_gee_full_coefficients.csv')
g.to_csv(o/'S2_Table_GEE_full_coefficients.csv',index=False)
fe=pd.read_csv(r/'results/05_fixed_effects/tables/24_full_coefficients.csv')
fe=fe[((fe.max_gap_hours==8)&fe.method.isin(['participant_FE_LPM','participant_day_FE_LPM']))|
      (fe.max_gap_hours==2)]
fe.to_csv(o/'S2_Table_fixed_effects_full_coefficients.csv',index=False)
if glmm_path.is_file():
 pd.read_csv(glmm_path).to_csv(o/'S2_Table_GLMM_full_coefficients.csv',index=False)
 pfit=r/'results/04_glmm/tables/21_testA_glmm_full_fit_diagnostics.csv'
 if pfit.is_file(): pd.read_csv(pfit).to_csv(o/'S2_Table_GLMM_fit_diagnostics.csv',index=False)
shutil.copyfile(r/'static/Table1_parent_study_aggregate.csv',o/'Table1_parent_study_aggregate_REFERENCE_ONLY.csv')
print('Manuscript Table 2, Table 1 aggregate reference, and complete model coefficient source tables:',o)
