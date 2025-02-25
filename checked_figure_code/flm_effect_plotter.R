flm_effect_plotter <- function(mainflm, sigdisp = TRUE) {
  require(cowplot)
  flm_data <- tibble()
  
  n_var <- nrow(mainflm$betaHat)
  
  for (r in 1:n_var) {
    if (is.null(mainflm$betaHat.var)) {
      beta.hat.plt <- data.frame(
        s = mainflm$argvals,
        beta = mainflm$betaHat[r, ]
      )
      beta.hat.plt |>
        as_tibble() |>
        mutate(var_num = r) |>
        bind_rows(flm_data) -> flm_data
    } else {
      beta.hat.plt <- data.frame(
        s = mainflm$argvals,
        beta = mainflm$betaHat[r, ],
        lower = mainflm$betaHat[r, ] - 2 * sqrt(diag(mainflm$betaHat.var[, , r])),
        upper = mainflm$betaHat[r, ] + 2 * sqrt(diag(mainflm$betaHat.var[, , r])),
        lower.joint = mainflm$betaHat[r, ] - mainflm$qn[r] * sqrt(diag(mainflm$betaHat.var[, , r])),
        upper.joint = mainflm$betaHat[r, ] + mainflm$qn[r] * sqrt(diag(mainflm$betaHat.var[, , r]))
      )
      beta.hat.plt |>
        as_tibble() |>
        mutate(var_num = r) |>
        bind_rows(flm_data) -> flm_data
    }
  }
  
  
  flm_data |>
    select(-lower, -upper) |>
    rowwise() |>
    mutate(cisig = if_else(between(0, lower.joint, upper.joint), 0, var_num * -0.1)) |>
    ungroup() |>
    select(s, beta, var_num, cisig) |>
    pivot_wider(names_from = var_num, values_from = c(beta, cisig)) |>
    rename(control = beta_1) |>
    select(-cisig_1) |>
    mutate(across(starts_with("beta"), ~ . + control)) -> flm_plot_data
  
  flm_plot_data |>
    select(starts_with("cisig"), s) |>
    pivot_longer(cols = !starts_with("s")) |>
    filter(value != 0) |>
    mutate(name = str_replace_all(name, "cisig", "beta")) -> sig_data
  
  flm_plot_data |>
    select(-starts_with("cisig")) |>
    pivot_longer(cols = !starts_with("s")) |>
    ggplot(aes(x = s, y = value, color = name)) +
    geom_line() +
    theme_cowplot() -> p
  
  if (sigdisp) {
    p + geom_point(data = sig_data, aes(x = s, y = value)) -> p
  }
  
  return(p)
}
