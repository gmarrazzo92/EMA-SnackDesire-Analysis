# Unresolved snack desire and subsequent snacking in daily life

This is the analysis code for our secondary analysis of the EMA monitoring study (136 participants, 21 days; Pimpini et al., 2026, *Eating Behaviors*, 62, 102115). It recreates the results, figures and statistical tables in the paper. The analyses were exploratory and were not part of the parent study's preregistration.

## Run it

1. Download the original study files from **https://osf.io/fsqpv/**. Put the downloaded ZIP in `data/raw/` and make sure the name is: `fsqpv-osfstorage-archive.zip`. You do **not** need to unpack it. Alternatively, put the ten CSV batch files in `data/raw/Raw_data/`.
2. Install the Python environment: `conda env create -f environment.yml`, then `conda activate ema-snack-reanalysis`.
3. Install R (we used R 4.6.0). Once, run `Rscript setup_R.R` to install any missing R packages.
4. Run `python run_all.py` from this folder. On Windows, if R is not found, run `python run_all.py --rscript "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"`.

The pipeline builds the reduced data first and runs the analyses in order. The complete run includes 5,000 surrogate permutations for each time-gap setting, so this step can be time consuming.

## Results are written to: 

- **Main figures:** `results/figures/` (Figures 1 and 2).
- **Supplementary Figure S1:** `results/peri_snack/S1_Fig_perisnack_trajectories_R.pdf` (also PNG).
- **Main Table 2 and full regression coefficients:** `results/publication_tables/`.
- **Supplementary Tables S1–S2:** `results/peri_snack/`.
- **Run log and numerical checks:** `results/run_log.txt`, `results/VALIDATION.json`, and `results/RUN_SUMMARY.json`.

Table 1 contains baseline participant characteristics from the parent study. Its aggregate values are included in `static/`. Supplementary Table S3 records questionnaire wording and is not produced from the numeric EMA files.

## Data and analysis details

`00_prepare_data.py` reads the ten public raw CSV batches, removes 21 near-duplicate prompts, and constructs the 16,732-row analysis dataset in `data/derived/`. It preserves all unanswered prompts, because skipping them would change which windows count as consecutive. It also preserves the original distinction between an explicit **No** (scored zero) and an unanswered follow-up (missing).

The prepared file uses pseudonymous participant labels and shifted dates. Its extra `original_matching_hour` column reproduces the clock-time transformation in the initial matched analysis. That historical transformation is retained so the controls match those used for the published results. The stricter pre-event matching sensitivity was corrected before submission; the public R script includes that correction.

Everything uses paths relative to this release folder. Input data and generated participant-level datasets are intentionally **not** bundled here; they come from the parent OSF study. If you want to use a different version of the parent data, you must first check that the reconstruction still gives the expected row counts and results.

The manuscript was checked using Python 3.11.17 (NumPy 2.4.6, pandas 3.0.6, statsmodels 0.15.0) and R 4.6.0 (lme4 2.0.1). `setup_R.R` installs missing packages and displays the versions it finds.

Questions about the code or analysis: contact the corresponding author listed in the paper.
