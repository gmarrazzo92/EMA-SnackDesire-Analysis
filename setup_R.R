# Run once using your installed R (R 4.6.0 used for the manuscript).
required <- c('dplyr','tidyr','purrr','tibble','readr','stringr','ggplot2','lubridate','lme4')
missing <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
if (length(missing)) {
  message('Installing R packages: ',paste(missing,collapse=', '))
  install.packages(missing,repos='https://cloud.r-project.org')
}
failed <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
if (length(failed)) stop('Missing R packages: ',paste(failed,collapse=', '))
message('R ',getRversion(),' — packages available:')
for (x in required) message('  ',x,' ',as.character(packageVersion(x)))
message('Manuscript GLMM was checked using R 4.6.0 and lme4 2.0.1.')
