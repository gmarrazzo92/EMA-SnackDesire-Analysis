"""Produce publication Figure 1 (study design) and Figure 2 (GEE and surrogates).

Reads the existing Test A analysis-pairs CSV plus the saved GEE/surrogate results.
Refits the *same* Test A GEE to estimate model-standardized 95% confidence
intervals by the delta method using the participant-cluster sandwich covariance.
No new study analyses or alternative model specifications are introduced.

Usage:
  python scripts/10_make_figures.py --results-root PATH_TO_EXTRACTED_RESULTS
  python scripts/10_make_figures.py --results-root PATH --output-dir FIGURES_PATH

PATH should contain 03_gee/ and 06_constrained_surrogate/.
Dependencies: numpy, pandas, matplotlib, scipy, patsy, statsmodels.
Outputs: PDF, SVG and PNG for Figures 1 and 2.
"""
from pathlib import Path
import argparse
from scipy.special import expit
from patsy import build_design_matrices, dmatrix
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results-root', type=Path, default=None, help='Release results directory containing 03_gee and 06_constrained_surrogate')
ap.add_argument('--output-dir', type=Path, default=None, help='Where figures are written (default: results/10_make_figures)')
args = ap.parse_args()
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = args.results_root or SCRIPT_DIR.parent/'results'
OUT = args.output_dir or ROOT/'10_make_figures'
OUT.mkdir(parents=True, exist_ok=True)
A = ROOT / '03_gee' / 'tables'
B = ROOT / '06_constrained_surrogate' / 'tables'
for required in (A/'21_testA_gee_results_by_gap.csv', A/'21_testA_analysis_pairs_8h.csv', B/'22_testB_conditional_shuffle_summary.csv', B/'22_testB_conditional_shuffle_iterations.csv'):
    if not required.exists():
        raise FileNotFoundError(f'Missing {required}. Run scripts 03 and 06 first or specify --results-root.')

COLORS = {
    'ink': '#273846', 'muted':'#566878', 'grid':'#D7DEE3', 'light':'#F1F5F7',
    'no':'#648394', 'yes':'#CF853D', 'null':'#7A9D9D',
    'out':'#78918B', 'white':'#FFFFFF', 'faint':'#F8FAFB',
}

mpl.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':9.2, 'axes.titlesize':11.5,
    'axes.titleweight':'bold', 'axes.labelsize':9, 'xtick.labelsize':8.2,
    'ytick.labelsize':8.2, 'savefig.facecolor':'white', 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'none', 'figure.facecolor':'white',
    'axes.spines.top':False, 'axes.spines.right':False,
})

def save(fig, name):
    for ext in ('pdf','svg','png'):
        opts = {'dpi': 350} if ext=='png' else {}
        fig.savefig(OUT/f'{name}.{ext}', bbox_inches='tight', pad_inches=.19, **opts)
    plt.close(fig)

def text(ax, x,y,s,**kw):
    ax.text(x,y,s,transform=ax.transAxes,**kw)

def panel_title(ax,letter,title,at_y=1.04):
    text(ax,-.01,at_y,letter,ha='left',va='bottom',fontsize=13,fontweight='bold',color=COLORS['ink'])
    text(ax,.055,at_y,title,ha='left',va='bottom',fontsize=10.6,fontweight='bold',color=COLORS['ink'])

def draw_box(ax, xy,wh,color,edge=None, rounding=.025):
    p=FancyBboxPatch(xy,wh[0],wh[1],boxstyle=f'round,pad=0.01,rounding_size={rounding}',
                     linewidth=1.0,facecolor=color,edgecolor=edge or color,transform=ax.transAxes)
    ax.add_patch(p)
    return p

# Fig 1: Schematic sampling positions on a clock-time scale; do not imply exact prescribed times for each dot.
fig=plt.figure(figsize=(9.2, 6.5))
grid=fig.add_gridspec(2,1,height_ratios=[.88,1.38],hspace=.12)
ax1=fig.add_subplot(grid[0]); ax1.axis('off');ax1.set_xlim(0,1);ax1.set_ylim(0,1)
panel_title(ax1,'A','EMA monitoring schedules')
text(ax1,.0,.92,'21 consecutive days  ·  identical EMA items  ·  notification times jittered ±5 min',
     fontsize=8.75,color=COLORS['muted'],va='center')
rows=[(3,49,.73),(6,44,.52),(9,43,.31)]
for n,subjects,y in rows:
    text(ax1,.01,y,f'{n} prompts/day',fontsize=9.3,color=COLORS['ink'],fontweight='normal',va='center')
    text(ax1,.18,y,f'(n = {subjects})',fontsize=8.3,color=COLORS['muted'],va='center')
    xa,xb=.39,.91
    ax1.plot([xa,xb],[y,y],lw=2.3,color=COLORS['grid'],solid_capstyle='round',transform=ax1.transAxes)
    ticks=np.linspace(xa,xb,n)
    ax1.scatter(ticks, [y]*n, s=55,color=COLORS['no'],edgecolors='white',linewidths=1.0,zorder=3,transform=ax1.transAxes)
# The landmarks of the former Figure 1, on a proportional 10:00–23:00 clock axis.
# Prompt dots are schematic / nominal; do not misrepresent the random ±5-minute jitter.
for hour, label in [(10,'10:00'), (13,'13:00'), (16.5,'16:30'), (20,'20:00'), (23,'23:00')]:
    x=.39 + (hour-10)/(23-10)*(.91-.39)
    ax1.plot([x,x],[.195,.22],color=COLORS['muted'],lw=.8,transform=ax1.transAxes)
    text(ax1,x,.155,label,ha='center',va='center',fontsize=8.3,color=COLORS['muted'])
text(ax1,.39,.075,'Illustrative nominal prompt positions; notifications jittered ±5 minutes.',
     fontsize=7.9,color=COLORS['muted'],va='center')

ax=fig.add_subplot(grid[1]); ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
panel_title(ax,'B','Primary across-interval comparison',at_y=1.01)
# two consecutive time intervals, parallel conditions
text(ax,.255,.9,'Reporting window t',ha='center',fontsize=9.4,color=COLORS['ink'],fontweight='bold')
text(ax,.70,.9,'Following window t + 1',ha='center',fontsize=9.4,color=COLORS['ink'],fontweight='bold')

# time-prompt ticks are shared across parallel conditions
for x, lab in [(.05,'Prompt t − 1'),(.45,'Prompt t'),(.95,'Prompt t + 1')]:
    ax.plot([x,x],[.25,.81],ls='--',lw=1.05,color=COLORS['grid'],transform=ax.transAxes,zorder=1)
    text(ax,x,.16,lab,ha='center',fontsize=8.25,color=COLORS['muted'])

# ranges
for y,desirecol,label in [(.7,COLORS['no'],'Desire absent'),(.41,COLORS['yes'],'Desire present')]:
    draw_box(ax,(.085,y-.095),(.32,.2),COLORS['light'],edge=COLORS['grid'],rounding=.025)
    ax.add_patch(Rectangle((.085,y-.095),.018,.2,facecolor=desirecol,edgecolor='none',transform=ax.transAxes,zorder=3))
    text(ax,.125,y+.045,label,fontsize=10.1,fontweight='bold',color=COLORS['ink'],va='center')
    text(ax,.125,y-.033,'No snack reported',fontsize=8.7,color=COLORS['muted'],va='center')
    ax.add_patch(FancyArrowPatch((.41,y),(.59,y),arrowstyle='-|>',mutation_scale=14,lw=1.8,color=COLORS['ink'],transform=ax.transAxes))
    draw_box(ax,(.60,y-.095),(.30,.2),'#F1F6F3',edge='#C5D5CF',rounding=.025)
    text(ax,.75,y+.045,'Any snacking?',ha='center',fontsize=10.1,color=COLORS['ink'],fontweight='bold',va='center')
    text(ax,.75,y-.033,'Yes / no',ha='center',fontsize=9.0,color=COLORS['muted'],va='center')
text(ax,.50,.07,'Next-window desire is not required for the outcome definition.',
     ha='center',fontsize=8.7,color=COLORS['muted'])
fig.subplots_adjust(left=.08,right=.98,top=.95,bottom=.06)
save(fig,'EMA_Figure1_StudyDesign')

# Figure 2: robust delta-method uncertainty for standardized marginal probabilities.
def gee_standardized_confidence_intervals(pairs_csv, saved_row):
    d = pd.read_csv(pairs_csv, low_memory=False)
    d['id'] = d['id'].astype(str)
    # Explicit Patsy design: do not rely on version-dependent locations of
    # design_info within statsmodels' formula model. Building the design once
    # also avoids the EvalEnvironment compatibility warning in newer versions.
    rhs = ('crave_within + crave_person_mean + '
           'C(next_group_slot) + study_day_c + log_gap')
    exog = dmatrix(rhs, data=d, return_type='dataframe')
    design = exog.design_info
    model = sm.GEE(endog=d['snack_next'], exog=exog, groups=d['id'],
                   family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
    # Guarantee the refitted model matches the published primary result.
    if not model.converged or not np.isclose(np.exp(model.params['crave_within']),
                                             float(saved_row.odds_ratio), atol=2e-6):
        raise RuntimeError('GEE refit differs from the saved Table 2 model. Check model data and versions.')
    columns = list(design.column_names)
    beta = model.params.loc[columns].to_numpy()
    cov = model.cov_params().loc[columns, columns].to_numpy()
    # Equal participant weight: each participant receives 1/n_ID,
    # shared equally across their own eligible reporting-window pairs.
    n_person = d.id.nunique()
    weights = (1.0 / n_person / d.groupby('id').id.transform('size')).to_numpy()
    out = {}
    for c in (0, 1):
        x = d.copy()
        x['crave_within'] = c - x['crave_person_mean']
        X = np.asarray(build_design_matrices([design], x)[0])
        predicted = expit(X @ beta)
        avg = float(weights @ predicted)
        grad = np.sum((weights * predicted * (1 - predicted))[:, None] * X, axis=0)
        se = float(np.sqrt(max(0, grad @ cov @ grad)))
        out[c] = {'p': avg, 'se': se, 'grad': grad,
                  'lo': max(0, avg - 1.95996398454*se),
                  'hi': min(1, avg + 1.95996398454*se)}
        saved = float(saved_row.std_probability_c0 if c==0 else saved_row.std_probability_c1)
        if not np.isclose(avg, saved, atol=2e-6):
            raise RuntimeError('Standardized predictions differ from the saved results')
    grad = out[1]['grad'] - out[0]['grad']
    delta_se = float(np.sqrt(max(0, grad @ cov @ grad)))
    diff = out[1]['p'] - out[0]['p']
    out['difference'] = {'p': diff, 'se': delta_se,
                         'lo': diff - 1.95996398454*delta_se,
                         'hi': diff + 1.95996398454*delta_se}
    return out

# Figure 2: model standardized probabilities and true permutation distribution.
gee=pd.read_csv(A/'21_testA_gee_results_by_gap.csv')
pr=gee.loc[np.isclose(gee.max_gap_h,8)].iloc[0]
ci=gee_standardized_confidence_intervals(A/'21_testA_analysis_pairs_8h.csv', pr)
shuffle=pd.read_csv(B/'22_testB_conditional_shuffle_iterations.csv')
summary=pd.read_csv(B/'22_testB_conditional_shuffle_summary.csv')
vals=shuffle.loc[np.isclose(shuffle.gap_h,8),'testA_raw_RD'].dropna().to_numpy()*100
null=summary.loc[np.isclose(summary.gap_h,8)&(summary.metric=='testA_raw_RD')].iloc[0]
assert len(vals)==5000, len(vals)
assert np.isclose(vals.mean(),null.null_mean*100)
assert np.isclose((float(pr.std_probability_c1)-float(pr.std_probability_c0))*100, float(pr.std_risk_difference)*100)

fig=plt.figure(figsize=(9.8,5.0))
grid=fig.add_gridspec(1,2,width_ratios=[.93,1.14],wspace=.22)

ax=fig.add_subplot(grid[0]);
for s in ['top','right','left']:
    ax.spines[s].set_visible(False)
ax.set_ylim(0,56);ax.set_xlim(-.7,1.7)
ax.set_yticks([0,10,20,30,40,50]);ax.set_yticklabels([f'{i}%' for i in [0,10,20,30,40,50]])
ax.grid(axis='y',lw=.65,color=COLORS['grid'],alpha=.8)
ax.set_axisbelow(True)
p=[float(pr.std_probability_c0)*100,float(pr.std_probability_c1)*100]
ax.bar([0,1],p,width=.54,color=[COLORS['no'],COLORS['yes']],edgecolor='none')
err_lower = [(ci[c]['p'] - ci[c]['lo'])*100 for c in (0,1)]
err_upper = [(ci[c]['hi'] - ci[c]['p'])*100 for c in (0,1)]
ax.errorbar([0,1],p,yerr=np.array([err_lower,err_upper]),fmt='none',
            ecolor=COLORS['ink'],elinewidth=1.5,capsize=5,capthick=1.4,zorder=4)
ax.set_xticks([0,1],['Desire absent','Desire present'])
ax.tick_params(axis='x',length=0,pad=9)
for i,v in enumerate(p):
    ax.text(i,ci[i]['hi']*100+1.15,f'{v:.1f}%',color=COLORS['ink'],ha='center',va='bottom',fontsize=12.0,fontweight='bold')
ax.set_ylabel('Probability of next-window snacking',color=COLORS['ink'])
ax.text(.5,54.0,f'+{(p[1]-p[0]):.1f} pp (95% CI {100*ci["difference"]["lo"]:.1f}–{100*ci["difference"]["hi"]:.1f})',ha='center',va='top',color=COLORS['ink'],fontsize=8.7,fontweight='bold')
panel_title(ax,'A','Model-standardized probabilities',at_y=1.075)


ax=fig.add_subplot(grid[1]);
for s in ['top','right']:
    ax.spines[s].set_visible(False)
bins=np.linspace(min(vals.min()-0.15,5.0),max(vals.max()+0.15,14.25),42)
ax.hist(vals,bins=bins,color=COLORS['null'],edgecolor='white',lw=.3,alpha=.94)
mean=float(null.null_mean)*100;obs=float(null.observed)*100
ci0=float(null.null_p025)*100;ci1=float(null.null_p975)*100
ax.axvspan(ci0,ci1,color=COLORS['null'],alpha=.115,zorder=0)
ax.axvline(mean,color=COLORS['ink'],lw=1.6,ls='--',zorder=4)
ax.axvline(obs,color=COLORS['yes'],lw=2.4,zorder=4)
ax.set_xlim(min(4.8,bins[0]),18.2)
ax.set_xlabel('Unadjusted next-window risk difference (pp)',labelpad=9,color=COLORS['ink'])
ax.set_ylabel('Surrogate datasets',color=COLORS['ink'])
ax.grid(axis='y',lw=.65,color=COLORS['grid'],alpha=.75)
ax.set_axisbelow(True)
ymax=ax.get_ylim()[1]
ax.set_ylim(0,ymax*1.08)
ymax=ax.get_ylim()[1]
ax.text(mean-0.1,ymax*.89,f'Surrogate mean\n{mean:.1f} pp',color=COLORS['ink'],fontsize=8.7,ha='right',va='top')
ax.text(obs-0.1,ymax*.89,f'Observed\n{obs:.1f} pp',color=COLORS['yes'],fontweight='bold',fontsize=9.0,ha='right',va='top')

panel_title(ax,'B','Constrained-surrogate reference distribution',at_y=1.075)


fig.subplots_adjust(left=.085,right=.985,bottom=.27,top=.84,wspace=.27)
fig.text(.085,.105, f'A  Participant-balanced GEE predictions; error bars: 95% CIs. '
                      f'Within-person OR {pr.odds_ratio:.2f} (95% CI {pr.OR_CI_low:.2f}–{pr.OR_CI_high:.2f}).',
         fontsize=8.25,color=COLORS['muted'],ha='left')
fig.text(.085,.061, f'B  5,000 constrained surrogates (empirical p = {float(null.p_upper):.4f}); '
                      'shading: central 95% surrogate range.',
         fontsize=8.25,color=COLORS['muted'],ha='left')
save(fig,'EMA_Figure2_ProspectiveResults')

print('FIGURE 1: schedules 3/day (n49), 6/day(n44), 9/day(n43); illustrative nominal times; axis 10:00–23:00')
print('FIGURE 2A: p0 %.2f%% CI[%.2f,%.2f]; p1 %.2f%% CI[%.2f,%.2f]; difference %.2f pp CI[%.2f,%.2f]' % (p[0],ci[0]['lo']*100,ci[0]['hi']*100,p[1],ci[1]['lo']*100,ci[1]['hi']*100,100*ci['difference']['p'],100*ci['difference']['lo'],100*ci['difference']['hi']))
print('FIGURE 2B: B=%s; mean=%.4fpp, observed=%.4fpp, central 95%%=[%.4f,%.4f]pp, emp p=%.6f'%(len(vals),mean,obs,ci0,ci1,float(null.p_upper)))
for f in sorted(OUT.glob('*')):
    print('FILE',f.name,f.stat().st_size)
