# Complete PLOS ONE GLMM export for EMA Test A (same fixed effects as original 21_testA_glmm.R).
# Usage:
#   Rscript scripts/04_glmm.R "results/03_gee/tables/21_testA_analysis_pairs_8h.csv"
# Outputs are aggregate statistics; the analysis-pairs data are not exported.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop('Supply the complete Test A analysis-pairs CSV path')
if (!requireNamespace('lme4', quietly = TRUE)) stop('Install lme4: install.packages("lme4")')
input_file <- normalizePath(args[1], mustWork = TRUE)
d <- read.csv(input_file, stringsAsFactors = FALSE)
required <- c('id','crave_t','snack_next','next_group_slot','day','gap_hours')
if (length(setdiff(required, names(d))) > 0) stop('Missing required data columns')
d <- d[complete.cases(d[, required]) & d$gap_hours > 0 & d$gap_hours <= 8, , drop=FALSE]
d$id <- factor(d$id)
d$next_group_slot <- factor(d$next_group_slot)
d$crave_t <- as.integer(d$crave_t)
d$snack_next <- as.integer(d$snack_next)
d$crave_person_mean <- ave(d$crave_t, d$id, FUN=mean)
d$crave_within <- d$crave_t - d$crave_person_mean
d$study_day_c <- d$day - mean(d$day)
d$log_gap <- log(d$gap_hours)

set.seed(20261009)
fit <- lme4::glmer(snack_next ~ crave_within + crave_person_mean + next_group_slot +
                   study_day_c + log_gap + (1|id), data=d,
                   family=binomial(link='logit'),
                   control=lme4::glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=2e5)))
summary_table <- coef(summary(fit))
if (!'crave_within' %in% rownames(summary_table)) stop('Missing primary coefficient')
beta <- summary_table[, 'Estimate']
se <- summary_table[, 'Std. Error']
fixed <- data.frame(
  model='GLMM_8h',
  term=rownames(summary_table),
  beta=as.numeric(beta),
  se=as.numeric(se),
  z=as.numeric(summary_table[,'z value']),
  p_value=as.numeric(summary_table[,'Pr(>|z|)']),
  beta_ci_low=as.numeric(beta-1.96*se),
  beta_ci_high=as.numeric(beta+1.96*se),
  odds_ratio=as.numeric(exp(beta)),
  OR_ci_low=as.numeric(exp(beta-1.96*se)),
  OR_ci_high=as.numeric(exp(beta+1.96*se)),
  n_pairs=nrow(d),
  n_participants=nlevels(d$id),
  row.names=NULL
)

random_variance <- as.numeric(lme4::VarCorr(fit)$id[1,1])
fit_info <- data.frame(
  model='GLMM_8h', n_pairs=nrow(d), n_participants=nlevels(d$id),
  random_intercept_variance=random_variance,
  random_intercept_sd=sqrt(random_variance),
  logLik=as.numeric(logLik(fit)), AIC=AIC(fit), BIC=BIC(fit),
  deviance=deviance(fit), n_fixed_effects=nrow(fixed),
  singular_fit=lme4::isSingular(fit),
  optimizer='bobyqa', approximation='Laplace',
  lme4_version=as.character(utils::packageVersion('lme4'))
)

# Keep GLMM outputs separate from GEE, even though they share its input pairs.
release_root <- Sys.getenv('EMA_RELEASE_ROOT', unset = '')
if (!nzchar(release_root)) {
  release_root <- normalizePath(file.path(dirname(input_file), '..', '..', '..'), mustWork = TRUE)
}
output_dir <- file.path(release_root, 'results', '04_glmm', 'tables')
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
fixed_path <- file.path(output_dir,'21_testA_glmm_full_fixed_effects.csv')
fit_path <- file.path(output_dir,'21_testA_glmm_full_fit_diagnostics.csv')
write.csv(fixed,fixed_path,row.names=FALSE)
write.csv(fit_info,fit_path,row.names=FALSE)
print(fit_info)
print(fixed[fixed$term=='crave_within',])
cat('Saved',fixed_path,'and',fit_path,'\n')
