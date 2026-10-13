# Matched peri-snack comparisons used for the published results.
# Primary matching uses participant/time information. The sensitivity analysis
# additionally weights preceding desire and low-control differences.
# Outputs are saved under results/02_match_perisnack/.

# Portable paths substituted below. Input RDS is rebuilt by
# scripts/01_prepare_matching_input.R from data/derived/ema_prompt_reduced.csv.
# Inputs are prepared by 01_prepare_matching_input.R.
# Control assignments are regenerated from the public OSF data.

rm(list = ls())

message("\n====================================================================")
message("RUNNING 02_match_perisnack")
message("Matched peri-snack controls: time-only primary + pre-event sensitivity.")
message("====================================================================\n")

# ---- USER SETTINGS ----------------------------------------------------------

# Path-independent: running from any working directory with EMA_RELEASE_ROOT set.
PROJECT_ROOT <- Sys.getenv("EMA_RELEASE_ROOT", unset = "")
if (!nzchar(PROJECT_ROOT)) stop("Run through run_all.py or set EMA_RELEASE_ROOT to the release root")
SEQ_OBJECT_FILE <- file.path(PROJECT_ROOT, "results", "01_prepare_matching_input", "matching_input.rds")
OUT_ROOT <- file.path(PROJECT_ROOT, "results", "02_match_perisnack")

OVERWRITE <- TRUE
SEED <- 20260630

# Event-locked display window.
# Core eligibility is t-1, t0, t+1; t-2 and t+2 are exported/plotted if present.
PERI_OFFSETS <- -2:2
CORE_OFFSETS <- c(-1L, 0L, 1L)

# Matching options.
# Primary analysis should usually use no control reuse.
ALLOW_CONTROL_REUSE_PRIMARY <- FALSE
ALLOW_CONTROL_REUSE_SENSITIVITY <- FALSE

# If no-reuse matching loses many snack events, this additional coverage check
# runs with reuse allowed. It should be treated as a coverage/sensitivity output,
# not the preferred primary analysis.
RUN_REUSE_COVERAGE_CHECK <- TRUE

# Candidate-control rules.
# Primary controls are required to have no snack at t0.
# The optional strict sensitivity requires no snack in t-1, t0, and t+1.
RUN_STRICT_NO_SNACK_CORE_SENSITIVITY <- TRUE

# Matching-score weights.
# Interpretation: 10 study days apart contributes one matching-score point;
# 50 points of t-1 CD difference contributes one point when pre-event matching is used.
MATCH_WEIGHT_DAY <- 0.10
MATCH_WEIGHT_PRE_CD <- 0.02
MATCH_WEIGHT_PRE_LOWCONTROL <- 0.02

# Install missing packages if needed.
INSTALL_MISSING_PACKAGES <- FALSE

# ---- PACKAGES ---------------------------------------------------------------

required_pkgs <- c(
  "dplyr", "tidyr", "purrr", "tibble", "readr", "stringr", "ggplot2", "lubridate"
)

missing_required <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_required) > 0) {
  if (INSTALL_MISSING_PACKAGES) {
    install.packages(missing_required)
  } else {
    stop("Missing required packages: ", paste(missing_required, collapse = ", "))
  }
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(lubridate)
})

set.seed(SEED)

select <- dplyr::select
filter <- dplyr::filter
lag <- dplyr::lag
lead <- dplyr::lead

# ---- OUTPUT FOLDERS ---------------------------------------------------------

if (!file.exists(SEQ_OBJECT_FILE)) {
  stop(
    "Missing reconstructed matching input RDS:\n",
    SEQ_OBJECT_FILE,
    "\nRun prepare_matching_input.R first."
  )
}

dir.create(OUT_ROOT, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_ROOT, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_ROOT, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_ROOT, "r_objects"), recursive = TRUE, showWarnings = FALSE)

OUT_TABLES <- file.path(OUT_ROOT, "tables")
OUT_FIGURES <- file.path(OUT_ROOT, "figures")
OUT_OBJECTS <- file.path(OUT_ROOT, "r_objects")

write_csv_safe <- function(x, filename) {
  path <- file.path(OUT_TABLES, filename)
  if (file.exists(path) && !OVERWRITE) {
    stop("Output exists and OVERWRITE = FALSE: ", path)
  }
  readr::write_csv(x, path, na = "")
  invisible(x)
}

save_plot_safe <- function(p, filename, width = 9, height = 5, dpi = 300) {
  path <- file.path(OUT_FIGURES, filename)
  if (file.exists(path) && !OVERWRITE) {
    stop("Output exists and OVERWRITE = FALSE: ", path)
  }
  ggplot2::ggsave(path, p, width = width, height = height, dpi = dpi)
  invisible(path)
}

safe_mean <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (sum(!is.na(x)) <= 1) NA_real_ else stats::sd(x, na.rm = TRUE)
}

safe_se <- function(x) {
  n <- sum(!is.na(x))
  if (n <= 1) return(NA_real_)
  stats::sd(x, na.rm = TRUE) / sqrt(n)
}

safe_ttest_onesample <- function(x, mu = 0) {
  x <- x[!is.na(x)]
  
  if (length(x) < 3 || stats::sd(x) == 0) {
    return(tibble(
      n = length(x),
      mean = safe_mean(x),
      sd = safe_sd(x),
      t = NA_real_,
      df = NA_real_,
      p = NA_real_,
      conf.low = NA_real_,
      conf.high = NA_real_
    ))
  }
  
  tt <- stats::t.test(x, mu = mu)
  
  tibble(
    n = length(x),
    mean = mean(x),
    sd = stats::sd(x),
    t = unname(tt$statistic),
    df = unname(tt$parameter),
    p = tt$p.value,
    conf.low = tt$conf.int[[1]],
    conf.high = tt$conf.int[[2]]
  )
}

offset_label_fun <- function(offset) {
  dplyr::case_when(
    offset < 0 ~ paste0("tm", abs(offset)),
    offset == 0 ~ "t0",
    offset > 0 ~ paste0("tp", offset)
  )
}

# ---- READ SEQUENCE-READY DATA ----------------------------------------------

prompt <- readRDS(SEQ_OBJECT_FILE)

required_cols <- c(
  "id", "group", "day", "timestamp", "segment_id", "segment_index", "segment_n",
  "hour_local", "current_snack_any", "CD", "LowControl", "Stress", "SA"
)

missing_cols <- setdiff(required_cols, names(prompt))

if (length(missing_cols) > 0) {
  stop(
    "The sequence-ready object is missing required columns:\n",
    paste(missing_cols, collapse = ", "),
    "\nRerun script 14 with the segmentation-fix version."
  )
}

if (!(".row_original" %in% names(prompt))) {
  prompt <- prompt %>% mutate(.row_original = row_number())
}

# Recompute local pre/post values inside segments to make this script standalone.
prompt <- prompt %>%
  arrange(id, segment_id, segment_index) %>%
  group_by(id, segment_id) %>%
  mutate(
    core_window_ok =
      segment_index >= (1L - min(CORE_OFFSETS)) &
      segment_index <= segment_n - max(CORE_OFFSETS),
    
    CD_m1 = dplyr::lag(CD),
    CD_p1 = dplyr::lead(CD),
    
    LowControl_m1 = dplyr::lag(LowControl),
    LowControl_p1 = dplyr::lead(LowControl),
    
    Stress_m1 = dplyr::lag(Stress),
    Stress_p1 = dplyr::lead(Stress),
    
    snack_m1 = dplyr::lag(current_snack_any),
    snack_p1 = dplyr::lead(current_snack_any),
    
    core_window_snack_count = {
      cs <- current_snack_any
      n_grp <- length(cs)
      
      purrr::map_int(segment_index, function(i) {
        idx <- i + CORE_OFFSETS
        idx <- idx[idx >= 1L & idx <= n_grp]
        sum(cs[idx] == 1, na.rm = TRUE)
      })
    }
  ) %>%
  ungroup()

# ---- BUILD SNACK EVENTS AND CONTROL CANDIDATES ------------------------------

snack_events <- prompt %>%
  filter(current_snack_any == 1, core_window_ok) %>%
  transmute(
    snack_focal_id = paste0("snack_", row_number()),
    focal_row = .row_original,
    id,
    group,
    day,
    segment_id,
    focal_segment_index = segment_index,
    focal_timestamp = timestamp,
    focal_hour = hour_local,
    
    # t0 values are outcomes/descriptives, not matching variables.
    focal_CD = CD,
    focal_LowControl = LowControl,
    focal_Stress = Stress,
    focal_SA = SA,
    
    # pre-event values may be used only in sensitivity matching.
    pre_CD = CD_m1,
    pre_LowControl = LowControl_m1,
    pre_Stress = Stress_m1
  )

control_candidates_all <- prompt %>%
  filter(current_snack_any == 0, core_window_ok) %>%
  transmute(
    control_focal_id = paste0("control_", row_number()),
    control_row = .row_original,
    id,
    group,
    day,
    segment_id,
    focal_segment_index = segment_index,
    focal_timestamp = timestamp,
    focal_hour = hour_local,
    
    focal_CD = CD,
    focal_LowControl = LowControl,
    focal_Stress = Stress,
    focal_SA = SA,
    
    pre_CD = CD_m1,
    pre_LowControl = LowControl_m1,
    pre_Stress = Stress_m1,
    
    core_window_snack_count = core_window_snack_count
  )

control_candidates_strict_no_snack <- control_candidates_all %>%
  filter(core_window_snack_count == 0)

write_csv_safe(snack_events, "15_snack_event_registry.csv")
write_csv_safe(control_candidates_all, "15_control_candidate_registry_all.csv")
write_csv_safe(control_candidates_strict_no_snack, "15_control_candidate_registry_strict_no_snack_core.csv")

# ---- MATCHING FUNCTION ------------------------------------------------------

match_controls <- function(
    snack_tbl,
    control_tbl,
    match_set,
    include_pre_event = FALSE,
    allow_reuse = FALSE
) {
  if (nrow(snack_tbl) == 0 || nrow(control_tbl) == 0) {
    return(tibble(
      match_set = character(),
      snack_focal_id = character(),
      control_focal_id = character(),
      id = character(),
      group = factor(character(), levels = levels(prompt$group)),
      snack_day = numeric(),
      control_day = numeric(),
      snack_hour = numeric(),
      control_hour = numeric(),
      snack_row = integer(),
      control_row = integer(),
      match_score = numeric(),
      day_distance = numeric(),
      hour_distance = numeric(),
      pre_CD_distance = numeric(),
      pre_LowControl_distance = numeric()
    ))
  }
  
  used_control_rows <- integer(0)
  match_list <- vector("list", nrow(snack_tbl))
  
  # Greedy matching. The ordering is deterministic and within-person.
  snack_tbl <- snack_tbl %>%
    arrange(id, day, focal_hour, focal_row)
  
  for (i in seq_len(nrow(snack_tbl))) {
    ev <- snack_tbl[i, ]
    
    cand <- control_tbl %>%
      filter(id == ev$id)
    
    if (!allow_reuse && length(used_control_rows) > 0) {
      cand <- cand %>%
        filter(!(control_row %in% used_control_rows))
    }
    
    if (include_pre_event) {
      # If using pre-event matching, require the pre-event variables to be available
      # for both event and candidate. We still do not match on t0 outcomes.
      cand <- cand %>%
        filter(
          !is.na(pre_CD),
          !is.na(pre_LowControl),
          !is.na(ev$pre_CD),
          !is.na(ev$pre_LowControl)
        )
    }
    
    if (nrow(cand) == 0) next
    
    cand <- cand %>%
      mutate(
        day_distance = abs(day - ev$day),
        hour_distance = abs(focal_hour - ev$focal_hour),
        pre_CD_distance = if (include_pre_event) abs(pre_CD - ev$pre_CD) else NA_real_,
        pre_LowControl_distance = if (include_pre_event) abs(pre_LowControl - ev$pre_LowControl) else NA_real_,
        
        match_score =
          hour_distance +
          MATCH_WEIGHT_DAY * day_distance +
          (if (include_pre_event) MATCH_WEIGHT_PRE_CD * pre_CD_distance else 0) +
          (if (include_pre_event) MATCH_WEIGHT_PRE_LOWCONTROL * pre_LowControl_distance else 0)
      ) %>%
      arrange(match_score, day_distance, hour_distance, control_row)
    
    best <- cand[1, ]
    
    used_control_rows <- c(used_control_rows, best$control_row)
    
    match_list[[i]] <- tibble(
      match_set = match_set,
      snack_focal_id = ev$snack_focal_id,
      control_focal_id = best$control_focal_id,
      id = ev$id,
      group = ev$group,
      
      snack_day = ev$day,
      control_day = best$day,
      
      snack_hour = ev$focal_hour,
      control_hour = best$focal_hour,
      
      snack_row = ev$focal_row,
      control_row = best$control_row,
      
      match_score = best$match_score,
      day_distance = best$day_distance,
      hour_distance = best$hour_distance,
      pre_CD_distance = best$pre_CD_distance,
      pre_LowControl_distance = best$pre_LowControl_distance
    )
  }
  
  out <- bind_rows(match_list)
  
  if (nrow(out) == 0) {
    out <- tibble(
      match_set = character(),
      snack_focal_id = character(),
      control_focal_id = character(),
      id = character(),
      group = factor(character(), levels = levels(prompt$group)),
      snack_day = numeric(),
      control_day = numeric(),
      snack_hour = numeric(),
      control_hour = numeric(),
      snack_row = integer(),
      control_row = integer(),
      match_score = numeric(),
      day_distance = numeric(),
      hour_distance = numeric(),
      pre_CD_distance = numeric(),
      pre_LowControl_distance = numeric()
    )
  }
  
  out
}

# ---- RUN MATCHING SETS ------------------------------------------------------

matched_primary <- match_controls(
  snack_tbl = snack_events,
  control_tbl = control_candidates_all,
  match_set = "primary_time_only_no_reuse",
  include_pre_event = FALSE,
  allow_reuse = ALLOW_CONTROL_REUSE_PRIMARY
)

matched_pre_event <- match_controls(
  snack_tbl = snack_events,
  control_tbl = control_candidates_all,
  match_set = "sensitivity_time_plus_tminus1_CD_LowControl_no_reuse",
  include_pre_event = TRUE,
  allow_reuse = ALLOW_CONTROL_REUSE_SENSITIVITY
)

matched_all <- bind_rows(matched_primary, matched_pre_event)

if (RUN_REUSE_COVERAGE_CHECK) {
  matched_primary_reuse <- match_controls(
    snack_tbl = snack_events,
    control_tbl = control_candidates_all,
    match_set = "coverage_time_only_reuse",
    include_pre_event = FALSE,
    allow_reuse = TRUE
  )
  
  matched_all <- bind_rows(matched_all, matched_primary_reuse)
}

if (RUN_STRICT_NO_SNACK_CORE_SENSITIVITY) {
  matched_strict <- match_controls(
    snack_tbl = snack_events,
    control_tbl = control_candidates_strict_no_snack,
    match_set = "sensitivity_time_only_strict_no_snack_core_no_reuse",
    include_pre_event = FALSE,
    allow_reuse = FALSE
  )
  
  matched_all <- bind_rows(matched_all, matched_strict)
}

write_csv_safe(matched_all, "15_matched_pairs_all.csv")

matching_counts <- matched_all %>%
  group_by(match_set) %>%
  summarise(
    snack_events_total = nrow(snack_events),
    controls_total_all = nrow(control_candidates_all),
    controls_total_strict_no_snack_core = nrow(control_candidates_strict_no_snack),
    matched_snack_events = n(),
    matched_participants = n_distinct(id),
    unique_control_windows = n_distinct(control_row),
    control_reuse_ratio = matched_snack_events / unique_control_windows,
    median_day_distance = median(day_distance, na.rm = TRUE),
    median_hour_distance = median(hour_distance, na.rm = TRUE),
    median_pre_CD_distance = median(pre_CD_distance, na.rm = TRUE),
    median_pre_LowControl_distance = median(pre_LowControl_distance, na.rm = TRUE),
    .groups = "drop"
  )

if (nrow(matching_counts) == 0) {
  matching_counts <- tibble(
    match_set = character(),
    snack_events_total = integer(),
    controls_total_all = integer(),
    controls_total_strict_no_snack_core = integer(),
    matched_snack_events = integer(),
    matched_participants = integer(),
    unique_control_windows = integer(),
    control_reuse_ratio = numeric(),
    median_day_distance = numeric(),
    median_hour_distance = numeric(),
    median_pre_CD_distance = numeric(),
    median_pre_LowControl_distance = numeric()
  )
}

write_csv_safe(matching_counts, "15_matching_counts.csv")

# ---- WINDOW EXTRACTION ------------------------------------------------------

make_windows <- function(focal_tbl, id_col_name, condition_label) {
  if (nrow(focal_tbl) == 0) {
    return(tibble(
      focal_id = character(),
      id = character(),
      group = factor(character(), levels = levels(prompt$group)),
      day = numeric(),
      segment_id = integer(),
      focal_segment_index = integer(),
      focal_timestamp = as.POSIXct(character()),
      focal_hour = numeric(),
      offset = integer(),
      target_segment_index = integer(),
      target_timestamp = as.POSIXct(character()),
      target_day = numeric(),
      target_hour = numeric(),
      CD = numeric(),
      LowControl = numeric(),
      Stress = numeric(),
      SA = numeric(),
      current_snack_any = integer(),
      condition = character()
    ))
  }
  
  focal_tbl %>%
    rename(focal_id = all_of(id_col_name)) %>%
    select(
      focal_id,
      id,
      group,
      day,
      segment_id,
      focal_segment_index,
      focal_timestamp,
      focal_hour
    ) %>%
    tidyr::crossing(offset = PERI_OFFSETS) %>%
    mutate(target_segment_index = focal_segment_index + offset) %>%
    left_join(
      prompt %>%
        transmute(
          id,
          segment_id,
          segment_index,
          target_timestamp = timestamp,
          target_day = day,
          target_hour = hour_local,
          CD,
          LowControl,
          Stress,
          SA,
          current_snack_any
        ),
      by = c("id", "segment_id", "target_segment_index" = "segment_index")
    ) %>%
    mutate(condition = condition_label)
}

snack_windows_all <- make_windows(
  snack_events %>% rename(focal_id_tmp = snack_focal_id),
  "focal_id_tmp",
  "snack_event"
)

control_windows_all <- make_windows(
  control_candidates_all %>% rename(focal_id_tmp = control_focal_id),
  "focal_id_tmp",
  "matched_non_snack"
)

# Build matched long data.
if (nrow(matched_all) > 0) {
  
  # Explicit join tables.
  # Do not rely on selecting the same column twice with rename syntax,
  # because dplyr may drop the original join column.
  snack_join_tbl <- matched_all %>%
    transmute(
      match_set = match_set,
      pair_id = paste(match_set, snack_focal_id, sep = "__"),
      focal_id = snack_focal_id
    )
  
  control_join_tbl <- matched_all %>%
    transmute(
      match_set = match_set,
      pair_id = paste(match_set, snack_focal_id, sep = "__"),
      focal_id = control_focal_id
    )
  
  snack_windows_matched <- snack_windows_all %>%
    inner_join(
      snack_join_tbl,
      by = "focal_id",
      relationship = "many-to-many"
    )
  
  control_windows_matched <- control_windows_all %>%
    inner_join(
      control_join_tbl,
      by = "focal_id",
      relationship = "many-to-many"
    )
  
  peri_long <- bind_rows(snack_windows_matched, control_windows_matched) %>%
    mutate(
      condition = factor(condition, levels = c("matched_non_snack", "snack_event")),
      offset = as.integer(offset)
    ) %>%
    pivot_longer(
      cols = c(CD, LowControl, Stress, SA, current_snack_any),
      names_to = "variable",
      values_to = "value"
    )
  
} else {
  
  peri_long <- tibble(
    focal_id = character(),
    id = character(),
    group = factor(character(), levels = levels(prompt$group)),
    day = numeric(),
    segment_id = integer(),
    focal_segment_index = integer(),
    focal_timestamp = as.POSIXct(character()),
    focal_hour = numeric(),
    offset = integer(),
    target_segment_index = integer(),
    target_timestamp = as.POSIXct(character()),
    target_day = numeric(),
    target_hour = numeric(),
    condition = factor(character(), levels = c("matched_non_snack", "snack_event")),
    match_set = character(),
    pair_id = character(),
    variable = character(),
    value = numeric()
  )
}

# ---- TRAJECTORY SUMMARIES ---------------------------------------------------

trajectory_summary <- peri_long %>%
  group_by(match_set, condition, variable, offset) %>%
  summarise(
    n_obs = sum(!is.na(value)),
    n_pairs = n_distinct(pair_id[!is.na(value)]),
    n_participants = n_distinct(id[!is.na(value)]),
    mean = mean(value, na.rm = TRUE),
    sd = stats::sd(value, na.rm = TRUE),
    se = safe_se(value),
    ci95_low = mean - 1.96 * se,
    ci95_high = mean + 1.96 * se,
    .groups = "drop"
  )

write_csv_safe(trajectory_summary, "15_perisnack_matched_trajectory_summary.csv")

# ---- OFFSET-LEVEL MATCHED CONTRASTS -----------------------------------------

offset_pair_contrast <- peri_long %>%
  select(match_set, pair_id, id, group, variable, offset, condition, value) %>%
  pivot_wider(names_from = condition, values_from = value) %>%
  mutate(
    contrast_snack_minus_control = snack_event - matched_non_snack
  )

write_csv_safe(offset_pair_contrast, "15_perisnack_offset_contrast_pair_level.csv")

offset_person_contrast <- offset_pair_contrast %>%
  group_by(match_set, id, variable, offset) %>%
  summarise(
    person_mean_contrast = mean(contrast_snack_minus_control, na.rm = TRUE),
    n_pairs_person = sum(!is.na(contrast_snack_minus_control)),
    .groups = "drop"
  )

write_csv_safe(offset_person_contrast, "15_perisnack_offset_contrast_person_level.csv")

offset_tests <- offset_person_contrast %>%
  group_by(match_set, variable, offset) %>%
  summarise(safe_ttest_onesample(person_mean_contrast), .groups = "drop") %>%
  group_by(match_set) %>%
  mutate(q_fdr_by_match_set = p.adjust(p, method = "BH")) %>%
  ungroup()

write_csv_safe(offset_tests, "15_perisnack_offset_tests_person_level.csv")

# ---- SHAPE-LEVEL MATCHED CONTRASTS ------------------------------------------

shape_event <- peri_long %>%
  filter(
    variable %in% c("CD", "LowControl", "Stress"),
    offset %in% CORE_OFFSETS
  ) %>%
  select(match_set, pair_id, id, group, condition, variable, offset, value) %>%
  mutate(offset_label = offset_label_fun(offset)) %>%
  select(-offset) %>%
  pivot_wider(names_from = offset_label, values_from = value) %>%
  mutate(
    rise_m1_to_0 = t0 - tm1,
    drop_0_to_p1 = tp1 - t0,
    peakness_0_vs_neighbors = t0 - ((tm1 + tp1) / 2)
  ) %>%
  select(
    match_set,
    pair_id,
    id,
    group,
    condition,
    variable,
    rise_m1_to_0,
    drop_0_to_p1,
    peakness_0_vs_neighbors
  ) %>%
  pivot_longer(
    cols = c(rise_m1_to_0, drop_0_to_p1, peakness_0_vs_neighbors),
    names_to = "shape_metric",
    values_to = "shape_value"
  )

write_csv_safe(shape_event, "15_perisnack_shape_event_level.csv")

shape_pair_contrast <- shape_event %>%
  pivot_wider(names_from = condition, values_from = shape_value) %>%
  mutate(
    contrast_snack_minus_control = snack_event - matched_non_snack
  )

write_csv_safe(shape_pair_contrast, "15_perisnack_shape_contrast_pair_level.csv")

shape_person_contrast <- shape_pair_contrast %>%
  group_by(match_set, id, variable, shape_metric) %>%
  summarise(
    person_mean_contrast = mean(contrast_snack_minus_control, na.rm = TRUE),
    n_pairs_person = sum(!is.na(contrast_snack_minus_control)),
    .groups = "drop"
  )

write_csv_safe(shape_person_contrast, "15_perisnack_shape_contrast_person_level.csv")

shape_tests <- shape_person_contrast %>%
  group_by(match_set, variable, shape_metric) %>%
  summarise(safe_ttest_onesample(person_mean_contrast), .groups = "drop") %>%
  group_by(match_set) %>%
  mutate(q_fdr_by_match_set = p.adjust(p, method = "BH")) %>%
  ungroup() %>%
  mutate(
    interpretation = dplyr::case_when(
      shape_metric == "rise_m1_to_0" & mean > 0 ~ "Snack windows show a larger pre-to-focal rise than matched non-snack windows.",
      shape_metric == "rise_m1_to_0" & mean < 0 ~ "Snack windows show a smaller pre-to-focal rise than matched non-snack windows.",
      shape_metric == "drop_0_to_p1" & mean < 0 ~ "Snack windows show a stronger post-focal drop than matched non-snack windows.",
      shape_metric == "drop_0_to_p1" & mean > 0 ~ "Snack windows show a weaker post-focal drop than matched non-snack windows.",
      shape_metric == "peakness_0_vs_neighbors" & mean > 0 ~ "Snack windows show stronger focal peakness than matched non-snack windows.",
      shape_metric == "peakness_0_vs_neighbors" & mean < 0 ~ "Snack windows show weaker focal peakness than matched non-snack windows.",
      TRUE ~ "Direction depends on metric sign."
    )
  )

write_csv_safe(shape_tests, "15_perisnack_shape_tests_person_level.csv")

# ---- BASIC QUALITY CHECK TABLE ----------------------------------------------

quality_check <- matching_counts %>%
  left_join(
    shape_tests %>%
      filter(variable == "CD", shape_metric == "peakness_0_vs_neighbors") %>%
      select(
        match_set,
        CD_peakness_contrast_mean = mean,
        CD_peakness_contrast_p = p,
        CD_peakness_contrast_q = q_fdr_by_match_set
      ),
    by = "match_set"
  ) %>%
  left_join(
    shape_tests %>%
      filter(variable == "LowControl", shape_metric == "peakness_0_vs_neighbors") %>%
      select(
        match_set,
        LowControl_peakness_contrast_mean = mean,
        LowControl_peakness_contrast_p = p,
        LowControl_peakness_contrast_q = q_fdr_by_match_set
      ),
    by = "match_set"
  )

write_csv_safe(quality_check, "15_quality_check_key_results.csv")

# ---- FIGURES ----------------------------------------------------------------

plot_trajectory_one_set <- function(match_set_name) {
  dat <- trajectory_summary %>%
    filter(
      match_set == match_set_name,
      variable %in% c("CD", "LowControl", "Stress")
    )
  
  if (nrow(dat) == 0) return(invisible(NULL))
  
  p <- dat %>%
    ggplot(aes(x = offset, y = mean, linetype = condition, group = condition)) +
    geom_ribbon(aes(ymin = ci95_low, ymax = ci95_high), alpha = 0.15, linetype = 0) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.8) +
    facet_wrap(~ variable, scales = "free_y") +
    scale_x_continuous(breaks = PERI_OFFSETS) +
    labs(
      title = paste0("Matched peri-snack trajectories: ", match_set_name),
      subtitle = "Snack windows versus within-person matched non-snack windows",
      x = "Prompt offset relative to focal window",
      y = "Mean value",
      linetype = "Window type"
    ) +
    theme_minimal(base_size = 12)
  
  filename <- paste0("15_fig_matched_trajectory_", stringr::str_replace_all(match_set_name, "[^A-Za-z0-9]+", "_"), ".png")
  save_plot_safe(p, filename, width = 10, height = 5.5)
}

plot_shape_contrast_one_set <- function(match_set_name) {
  dat <- shape_tests %>%
    filter(
      match_set == match_set_name,
      variable %in% c("CD", "LowControl", "Stress")
    )
  
  if (nrow(dat) == 0) return(invisible(NULL))
  
  p <- dat %>%
    ggplot(aes(x = shape_metric, y = mean)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15) +
    facet_wrap(~ variable, scales = "free_y") +
    labs(
      title = paste0("Matched peri-snack shape contrasts: ", match_set_name),
      subtitle = "Positive = snack greater than matched control, except drop where more negative = stronger drop",
      x = "Shape metric",
      y = "Snack minus matched-control contrast"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  
  filename <- paste0("15_fig_shape_contrast_", stringr::str_replace_all(match_set_name, "[^A-Za-z0-9]+", "_"), ".png")
  save_plot_safe(p, filename, width = 10, height = 5.5)
}

for (ms in unique(matched_all$match_set)) {
  plot_trajectory_one_set(ms)
  plot_shape_contrast_one_set(ms)
}

# ---- INTERPRETATION KEY -----------------------------------------------------

interpretation_key <- tribble(
  ~file, ~what_it_tests, ~how_to_interpret,
  
  "15_matching_counts.csv",
  "Counts matched snack events, controls, unique controls, and match quality by matching set.",
  "Inspect first. Primary no-reuse matching should retain enough events; reuse version is only a coverage sensitivity.",
  
  "15_quality_check_key_results.csv",
  "Compact table with matching counts and key CD/LowControl peakness contrasts.",
  "Fastest table for deciding whether the matched peri-snack comparison supports the story.",
  
  "15_matched_pairs_all.csv",
  "All matched snack-control pairs for all matching specifications.",
  "Use to audit day/hour distances and whether controls were reused.",
  
  "15_perisnack_matched_trajectory_summary.csv",
  "Mean trajectories for snack and matched non-snack windows from t-2 to t+2.",
  "Use for the main peri-snack figure if the primary matching set is adequate.",
  
  "15_perisnack_offset_tests_person_level.csv",
  "Person-level tests of snack-minus-control differences at each offset.",
  "Shows whether snack windows differ from controls at t-1, t0, and t+1.",
  
  "15_perisnack_shape_tests_person_level.csv",
  "Main matched peri-snack shape tests: rise, drop, and peakness contrasts.",
  "This is the key validation test. For CD and LowControl, expected pattern is larger rise, stronger negative drop, and stronger peakness for snack windows.",
  
  "15_fig_matched_trajectory_<match_set>.png",
  "Trajectory plot for each matching set.",
  "Best visual test of whether the snack event is a stronger local peak than matched non-snack windows.",
  
  "15_fig_shape_contrast_<match_set>.png",
  "Shape-contrast plot for each matching set.",
  "Positive peakness and rise, plus negative drop, support a snack-specific transient peak."
)

write_csv_safe(interpretation_key, "15_interpretation_key.csv")

# ---- SAVE CORE OBJECTS ------------------------------------------------------

saveRDS(
  list(
    prompt = prompt,
    snack_events = snack_events,
    control_candidates_all = control_candidates_all,
    control_candidates_strict_no_snack = control_candidates_strict_no_snack,
    matched_all = matched_all,
    peri_long = peri_long,
    trajectory_summary = trajectory_summary,
    offset_tests = offset_tests,
    shape_tests = shape_tests,
    quality_check = quality_check
  ),
  file.path(OUT_OBJECTS, "15_matched_perisnack_results.rds")
)

# ---- FINISH ----------------------------------------------------------------

message("\n====================================================================")
message("DONE: 02_match_perisnack")
message("Outputs written to:")
message(OUT_ROOT)

message("\nInspect these first:")
message(" - 15_matching_counts.csv")
message(" - 15_quality_check_key_results.csv")
message(" - 15_perisnack_shape_tests_person_level.csv")
message(" - 15_perisnack_matched_trajectory_summary.csv")
message(" - 15_perisnack_offset_tests_person_level.csv")
message(" - 15_fig_matched_trajectory_primary_time_only_no_reuse.png")
message(" - 15_fig_shape_contrast_primary_time_only_no_reuse.png")
message("====================================================================\n")