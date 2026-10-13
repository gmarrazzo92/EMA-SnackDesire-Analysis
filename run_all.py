#!/usr/bin/env python3
"""Reproduce the EMA snack-desire analyses from the public parent-study data.

From the release root, run:
    python run_all.py
or, if Rscript is not on your PATH:
    python run_all.py --rscript "C:\\Program Files\\R\\R-4.6.0\\bin\\Rscript.exe"
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'


def locate_r(user_arg: str | None) -> str:
    if user_arg:
        p = Path(user_arg).expanduser()
        if p.is_file(): return str(p.resolve())
        found = shutil.which(user_arg)
        if found: return found
        raise SystemExit(f'Rscript not found: {user_arg}')
    found = shutil.which('Rscript')
    if found: return found
    if os.name == 'nt':
        base = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'R'
        choices = sorted(base.glob('R-*/bin/Rscript.exe'),reverse=True)
        if choices: return str(choices[0])
    raise SystemExit('Rscript not found. Install R and run setup_R.R, then pass '
                     '--rscript PATH if needed. See README.md.')


def main() -> None:
    ap = argparse.ArgumentParser(description='Regenerate all manuscript and supplement analyses.')
    ap.add_argument('--rscript', help='Location of Rscript executable, if not on PATH')
    a = ap.parse_args()
    rscript = locate_r(a.rscript)
    RES.mkdir(exist_ok=True)
    log = RES / 'run_log.txt'
    log.write_text('EMA snack desire reproducibility run\n',encoding='utf-8')
    py=sys.executable
    scripts=ROOT / 'scripts'
    env=os.environ.copy();env['EMA_RELEASE_ROOT']=str(ROOT)
    history=[]

    def execute(name: str, cmd: list[str|Path]) -> None:
        print(f'\n=== {name} ===',flush=True)
        args=[str(x) for x in cmd]
        proc=subprocess.run(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,errors='replace')
        with log.open('a',encoding='utf-8') as fh:
            fh.write(f'\n=== {name} ===\n{args}\n{proc.stdout}')
        print(proc.stdout[-2200:],flush=True)
        history.append({'step':name,'status':'PASS' if proc.returncode==0 else 'FAIL',
                        'exit_code':proc.returncode})
        if proc.returncode:
            (RES/'RUN_SUMMARY.json').write_text(json.dumps({'status':'FAIL','steps':history},indent=2))
            raise SystemExit(f'{name} failed. Inspect results/run_log.txt.')

    execute('00 Prepare reduced data from original OSF raw files',[py,scripts/'00_prepare_data.py'])
    execute('01 Prepare peri-snack matching inputs',[rscript,scripts/'01_prepare_matching_input.R'])
    execute('02 Corrected matched peri-snack analyses',[rscript,scripts/'02_match_perisnack.R'])
    matches=RES/'02_match_perisnack/tables/15_matched_pairs_all.csv'
    execute('03 Logistic GEE models',[py,scripts/'03_gee.py','--project-root',ROOT])
    pairs=RES/'03_gee/tables/21_testA_analysis_pairs_8h.csv'
    execute('04 Random-intercept GLMM',[rscript,scripts/'04_glmm.R',pairs])
    execute('05 Participant and participant-day fixed effects',[py,scripts/'05_fixed_effects.py','--project-root',ROOT])
    execute('06 Constrained surrogate: 5,000 permutations per interval threshold',
            [py,scripts/'06_constrained_surrogate.py','--project-root',ROOT,'--B','5000'])
    execute('07 Matched preceding-window desire',[py,scripts/'07_preceding_desire.py',
            '--project-root',ROOT,'--matched-pairs',matches])
    execute('08 Supplementary tables and peri-snack source data',[py,scripts/'08_perisnack_tables.py',
            '--project-root',ROOT,'--matched-pairs',matches])
    execute('09 Main manuscript tables and full regression coefficients',[py,scripts/'09_build_tables.py',
            '--project-root',ROOT])
    execute('10 Main manuscript figures',[py,scripts/'10_make_figures.py',
            '--results-root',RES,'--output-dir',RES/'10_make_figures'])
    execute('11 Supplementary figure S1 (original R style)',[rscript,scripts/'11_make_S1_figure.R'])
    summary={'status':'PASS','source':'Original parent-study OSF raw batch exports',
             'rscript':rscript,'permutations_per_gap':5000,'steps':history}
    (RES/'RUN_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nALL STEPS PASSED. See results/09_build_tables, results/10_make_figures and results/11_make_S1_figure.')

if __name__=='__main__':main()
