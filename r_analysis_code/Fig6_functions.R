
new_label_data <- function(data) {
  require(tidyverse)
  
  data |>
    group_by(response_index, condition, test, seed, trained, naive) |>
    summarise(xs1 = sum(X1), xs2 = sum(X2), xs3 = sum(X3), xs4 = sum(X4)) |>
    mutate(t_type = case_when(
      xs1 == 1 & xs4 == 1 ~ "Cue A Rewarded",
      xs1 == 1 & xs4 == 0 ~ "Cue A Unrewarded",
      xs3 == 1 & xs4 == 1 ~ "Cue C Rewarded",
      xs3 == 1 & xs4 == 0 ~ "Cue C Unrewarded",
      xs2 == 1 ~ "Cue B",
      xs1 == 0 & xs4 == 1 ~ "Degradation",
      xs1 == 0 & xs2 == 0 & xs4 == 0 ~ "Blank"
    )) |>
    right_join(data, join_by(response_index, condition, test, seed, trained, naive)) |>
    select(-starts_with("xs")) |>
    ungroup() -> data
  return(data)
}

naive_vs_pretrained <- function(data) {
  data |> mutate(naive_group = case_when(
    condition == "conditioning" ~ TRUE,
    condition != "conditioning" & naive == TRUE ~ TRUE,
    .default = FALSE
  ), pretrained_group = case_when(
    condition == "conditioning" ~ TRUE,
    condition != "conditioning" & naive == FALSE ~ TRUE,
    .default = FALSE
  )) -> return_data
  return(return_data)
}

new_align_time <- function(data) {
  data |>
    # Find the minimum time for each trial where X1, X2, or X3 is 1
    group_by(response_index, condition, test, seed, trained, naive) |>
    mutate(first_event_time = ifelse(any(X1 == 1 | X2 == 1 | X3 == 1 | X4 == 1),
                                     min(t[X1 == 1 | X2 == 1 | X3 == 1 | X4 == 1], na.rm = TRUE),
                                     NA_real_
    )) |>
    mutate(first_event_time = ifelse(is.na(first_event_time), min(t), first_event_time)) |>
    # Create aligned time variable
    mutate(aligned_time = t - first_event_time) |>
    ungroup() |>
    mutate(aligned_time = case_when(
      t_type == "Degradation" ~ aligned_time + 9,
      t_type != "Degradation" ~ aligned_time
    )) -> data
  return(data)
}

mega_cca <- function(data, ndim = 3, reg = 0., threshold = 0.95) {
  data |>
    select_if(~ sum(!is.na(.)) > 0) |>
    ungroup() -> data
  
  cond <- data |>
    filter(condition == "conditioning", test == FALSE, trained == TRUE, )
  cuec <- data |>
    filter(condition == "cue-c", test == FALSE, trained == TRUE)
  degrade <- data |>
    filter(condition == "degradation", test == FALSE, trained == TRUE)
  
  X1 <- do_pca_norm(cond, threshold = threshold)
  X2 <- do_pca_norm(cuec, threshold = threshold)
  X3 <- do_pca_norm(degrade, threshold = threshold)
  
  min_ncol <- min(ncol(X1$pc_matrix), ncol(X2$pc_matrix), ncol(X3$pc_matrix))
  
  res.cc <- run_cca_r(X1$pc_matrix[, 1:min_ncol], X2$pc_matrix[, 1:min_ncol], X3$pc_matrix[, 1:min_ncol], ndim, reg)
  
  cond_all <- data |>
    filter(condition == "conditioning") |>
    compute_cc_scores(coef = (X1$pc_matrix[, 1:min_ncol] %*% res.cc$X_coeff))
  cuec_all <- data |>
    filter(condition == "cue-c") |>
    compute_cc_scores(coef = (X2$pc_matrix[, 1:min_ncol] %*% res.cc$Y_coeff))
  degrade_all <- data |>
    filter(condition == "degradation") |>
    compute_cc_scores(coef = (X3$pc_matrix[, 1:min_ncol] %*% res.cc$Z_coeff))
  
  return_data <- bind_rows(cond_all, cuec_all, degrade_all)
  return(return_data)
}

run_cca_r <- function(X, Y, Z, numCC = 3, reg = 0) {
  # Call the Python function
  coeffs <- run_cca(X, Y, Z, numCC, reg)
  
  # Convert Python results to R list
  list(
    X_coeff = matrix(unlist(coeffs[[1]]), ncol = numCC),
    Y_coeff = matrix(unlist(coeffs[[2]]), ncol = numCC),
    Z_coeff = matrix(unlist(coeffs[[3]]), ncol = numCC)
  )
}

do_pca_norm <- function(data, threshold = 0.95) {
  # Filter columns that start with 'Z'
  z_data <- data |> select(starts_with("Z"))
  
  # Run PCA
  pca_prep <- z_data |>
    recipe(~.) |>
    step_center(all_predictors()) |>
    step_scale(all_predictors()) |>
    step_pca(all_predictors(), threshold = threshold) |>
    prep()
  
  pca_res <- pca_prep |> bake(new_data = NULL)
  
  n_comp <- ncol(pca_res)
  
  pc_matrix <- tidy(pca_prep, number = 3) |>
    filter(parse_number(component) < (n_comp + 1)) |>
    select(-id) |>
    pivot_wider(id_cols = terms, names_from = component) |>
    select(-terms) |>
    as.matrix()
  
  # Return original values, PCA loadings and PCA values
  return_list <- list(
    data = bind_cols(data, pca_res),
    pc_matrix = pc_matrix
  )
  return(return_list)
}

compute_cc_scores <- function(data, coef) {
  data |>
    select(starts_with("Z")) |>
    mutate_all(scale) |>
    as.matrix() -> Z_M
  Z_M %*% coef -> cca_M
  colnames(cca_M) <- paste0("V", seq(1, ncol(cca_M)))
  
  data |> bind_cols(as_tibble(cca_M)) -> return_data
  return(return_data)
}

shift_iti <- function(data) {
  data |>
    group_by(trained, seed, test, naive, condition) |>
    mutate(iti = aligned_time < (-4)) |>
    mutate(response_index = response_index - as.numeric(iti)) |>
    new_align_time() |>
    filter(response_index > (-1)) |>
    group_by(response_index, trained, seed, test, naive, condition, pca_group) |>
    mutate(t_type = t_type[aligned_time == min(aligned_time)]) |>
    ungroup() -> return_data
  return(return_data)
}