# Construct the exact variables consumed by the original 15 matched-perisnack
# algorithm from the minimal prompt CSV (no obsolete Markov/surrogate analysis).
# CRITICAL: original_matching_hour was calculated BEFORE privacy date-shifting,
# using the same UTC -> Europe/Amsterdam hour conversion as original script 14.
# Do not recompute clock hour from scheduled_datetime (dates are now shifted,
# which destroys the original daylight-saving offsets).
# Called from run_all.py with EMA_RELEASE_ROOT set to the release directory.
root <- Sys.getenv('EMA_RELEASE_ROOT', unset='')
if (!nzchar(root)) stop('EMA_RELEASE_ROOT is not set; use run_all.py')
for (pkg in c('dplyr','readr','lubridate')) if (!requireNamespace(pkg,quietly=TRUE)) stop('Missing R package: ',pkg)
suppressPackageStartupMessages({library(dplyr);library(readr);library(lubridate)})
f <- file.path(root,'data','derived','ema_prompt_reduced.csv')
d <- readr::read_csv(f,show_col_types=FALSE,guess_max=100000)
required <- c('ID','Group','scheduled_datetime','StudyDay','CD','Snacked_YN','SA','Self_Control','Stress','original_matching_hour')
if (length(setdiff(required,names(d)))) stop('Reduced EMA file is missing: ',paste(setdiff(required,names(d)),collapse=', '))
parse_snack <- function(x,amount) {
  txt <- tolower(trimws(as.character(x)))
  out <- ifelse(txt %in% c('yes','y','true','1','si','sí','snacked'),1L,
                ifelse(txt %in% c('no','n','false','0'),0L,NA_integer_))
  fallback <- is.na(out) & !is.na(amount)
  out[fallback] <- as.integer(amount[fallback]>0)
  out
}
# In the source pipeline, .row_original was assigned before chronological sorting;
# its indices identify the original event and matched control windows.
prompt <- d %>% mutate(.row_original=row_number()) %>%
  transmute(id=as.character(ID), group=factor(as.character(Group),levels=c('Low','Med','High')),
    timestamp=as.POSIXct(scheduled_datetime,format='%Y-%m-%d %H:%M:%S',tz='UTC'),
    day=as.integer(StudyDay),CD=as.numeric(CD),SA=as.numeric(SA),
    Stress=as.numeric(Stress),Self_Control=as.numeric(Self_Control),
    original_matching_hour=as.numeric(original_matching_hour),
    LowControl=100-as.numeric(Self_Control),
    current_snack_any=parse_snack(Snacked_YN,as.numeric(SA)),
    .row_original=.row_original) %>%
  filter(!is.na(id),!is.na(group),!is.na(timestamp)) %>%
  arrange(id,timestamp,.row_original) %>% group_by(id) %>%
  mutate(gap_since_previous_hours=as.numeric(difftime(timestamp,lag(timestamp),units='hours')),
         new_segment=is.na(gap_since_previous_hours)|gap_since_previous_hours<=0|gap_since_previous_hours>8,
         segment_id=cumsum(new_segment)) %>% ungroup() %>%
  group_by(id,segment_id) %>%
  mutate(segment_index=row_number(),segment_n=n(),
      hour_local=as.numeric(original_matching_hour),
      core_window_ok=segment_index>=2L & segment_index<=segment_n-1L,
      core_window_snack_count=vapply(segment_index,function(i){
         idx=seq.int(max(1L,i-1L),min(segment_n[1],i+1L));sum(current_snack_any[idx]==1L,na.rm=TRUE)
      },integer(1))) %>% ungroup()
stopifnot(nrow(prompt)==16732)
out <- file.path(root,'results','01_prepare_matching_input');dir.create(out,recursive=TRUE,showWarnings=FALSE)
saveRDS(prompt,file.path(out,'matching_input.rds'))
cat('Prepared original matching-script input:', nrow(prompt),'rows\n')
