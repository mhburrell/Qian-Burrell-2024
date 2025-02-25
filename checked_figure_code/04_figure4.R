# Figure 4
# For experimental data, use 00_data_import.R

# For model data, see 03_figure3.R
sim_summary <- read_parquet("td_sim_summary_results.parquet")

# Panel C

# Experimental Data

lick_data |>
  left_join(mouse_info) |>
  filter(day %in% c(4, 9), trialtype %in% c("no_go")) |>
  filter(!(day == 9 & group == "conditioning")) |>
  mutate(group = if_else(day == 4, "conditioning", group)) |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  mutate(lick_bin = cut(lick_offset, breaks = seq(-2, 8, by = 0.2), labels = seq(-1.9, 7.9, 0.2))) |>
  filter(!is.na(lick_bin)) |>
  group_by(mouse, day, trialtype, lick_bin, group) |>
  reframe(lick_rate = 5 * n() / mean(n_trials)) |>
  group_by(group, trialtype, lick_bin, day) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  mutate(plot_group = "A") |>
  ggplot() +
  aes(as.numeric(as.character(lick_bin)), mean_lick_rate, color = plot_group, ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA, aes(group = plot_group)) +
  scale_x_continuous(breaks = seq(-2, 6, by = 2)) +
  scale_y_continuous(breaks = c(0, 1)) +
  facet_grid(group ~ .) +
  theme_cowplot() +
  theme(legend.position = "top") +
  xlab("Time (s)") +
  ylab("Lick rate (/s)")

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSm", model_switch %in% c(2, 4)) |>
  group_by(trialtype, testgroup, model_switch, t_rel) |>
  reframe(value = mean(value), rpe = mean(rpe)) |>
  ggplot() +
  aes(t_rel, value, color = factor(testgroup)) +
  facet_grid(testgroup ~ model_switch) +
  geom_line() +
  xlim(0, 30)

# Panel D

lick_data |>
  left_join(mouse_info) |>
  filter(day %in% c(4, 9), trialtype %in% c("no_go")) |>
  filter(!(day == 9 & group == "conditioning")) |>
  mutate(group = if_else(day == 4, "conditioning", group)) |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  mutate(time_period = case_when(
    between(lick_offset, 3.5, 5) ~ "early",
    between(lick_offset, 7, 8) ~ "late",
    .default = "outside"
  )) |>
  group_by(mouse, day, trialtype, time_period, group) |>
  reframe(lick_rate = n() / mean(n_trials)) |>
  mutate(lick_rate = if_else(time_period == "early", lick_rate * 1 / 1.5, lick_rate)) |>
  pivot_wider(names_from = time_period, values_from = lick_rate) |>
  mutate(across(c(early, late), ~ replace_na(., 0))) |>
  select(-outside) |>
  pivot_longer(c(early, late), names_to = "time_period", values_to = "lick_rate") -> data_by_mouse


data_by_mouse |>
  group_by(time_period, group) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  ggplot() +
  aes(time_period, mean_lick_rate, fill = group) +
  facet_grid(. ~ group) +
  geom_col() +
  geom_errorbar(width = 0.2, aes(ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate)) +
  geom_point(data = data_by_mouse, aes(time_period, lick_rate, color = group), shape = 1) +
  geom_line(data = data_by_mouse, aes(time_period, lick_rate, group = mouse), color = "grey") +
  theme_cowplot() +
  theme(legend.position = "none") -> figd_upper

# statistics

data_by_mouse |>
  pivot_wider(names_from = time_period, values_from = lick_rate) -> data_for_stats

data_for_stats |> filter(group == "conditioning") -> cond_data
t.test(cond_data$early, cond_data$late, paired = T) |> broom::tidy()

data_for_stats |> filter(group == "degradation") -> deg_data
t.test(deg_data$early, deg_data$late, paired = T) |> broom::tidy()

data_for_stats |> filter(group == "cuedrew") -> cr_data
t.test(cr_data$early, cr_data$late, paired = T) |> broom::tidy()

# Model summary

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSm", model_switch %in% c(2, 4)) |>
  mutate(time_period = case_when(
    between(t_rel, 3.5 * 5, 5 * 5) ~ "early",
    between(t_rel, 7 * 5, 8 * 5) ~ "late",
    .default = "outside"
  )) |>
  filter(time_period != "outside") |>
  group_by(testgroup, model_switch, rep, time_period) |>
  reframe(value = mean(value)) |>
  group_by(testgroup, model_switch, time_period) |>
  reframe(meanvalue = mean(value), sdvalue = sd(value)) |>
  ggplot() +
  aes(time_period, meanvalue, fill = factor(testgroup), ymin = meanvalue - sdvalue, ymax = meanvalue + sdvalue) +
  facet_grid(. ~ model_switch + testgroup) +
  geom_col() +
  geom_errorbar(width = 0.2) +
  theme_cowplot() +
  theme(legend.position = "none") -> figd_lower

# Panel F

labelled_photo_data |>
  filter(day %in% c(3, 4, 8, 9), trialtype %in% c("go", "go_omit"), between(time, 40, 60), anticipatory_licks > 0) |>
  group_by(day, mouse, group, trialnumber, pre_licks) |>
  reframe(max_da = max(data)) |>
  mutate(group = if_else(day < 5, "conditioning", group)) -> panel5_data

# bin data for display purposes only
panel5_data |>
  group_by(mouse, group) |>
  mutate(pre_licks = round(pre_licks / 2, 0)) |>
  group_by(mouse, group, pre_licks) |>
  reframe(max_da = mean(max_da)) -> panel5_data_binned


panel5_data |>
  ggplot() +
  aes(pre_licks / 2, max_da, colour = mouse) +
  facet_grid(~group) +
  xlim(0, 8) +
  geom_smooth(method = "lm", se = F) +
  geom_point(data = panel5_data_binned, aes(pre_licks, max_da, color = mouse)) +
  theme_cowplot() +
  theme(legend.position = "none")

# Panel G

# experimental data
# for each mouse, group do a linear regression of max_da on pre_licks

panel5_data |>
  group_by(mouse, group) |>
  nest() |>
  mutate(model = map(data, ~ lm(max_da ~ pre_licks, data = .x))) |>
  mutate(tidy = map(model, broom::tidy)) |>
  unnest(tidy) |>
  filter(term == "pre_licks") -> panel5_summarized

panel5_summarized |>
  ggplot() +
  aes(group, estimate, color = group) +
  geom_boxplot() +
  geom_point(shape = 1)

# stats
panel5_summarized |>
  filter(group == "conditioning") -> cond_data
# one-sample t-test, difference from zero
t.test(cond_data$estimate, mu = 0) |> broom::tidy()

panel5_summarized |>
  filter(group == "degradation") -> deg_data
t.test(deg_data$estimate, mu = 0) |> broom::tidy()

panel5_summarized |>
  filter(group == "cuedrew") -> cr_data
t.test(cr_data$estimate, mu = 0) |> broom::tidy()

# Model

require(tidyverse)
require(arrow)
require(duckdb)

# model labels 1: CSC (no ITI), 2: Cue-Context, 3: CSC with ITI, 4: Belief State

duck_con <- dbConnect(duckdb::duckdb(dbdir = "./duck1.db"))
dbExecute(duck_con, "PRAGMA memory_limit='40GB'")

# dbExecute(duck_con, "CREATE VIEW results AS SELECT * FROM read_parquet('./sim_data/m1_3_results/*.parquet')")
results <- tbl(duck_con, "results")

# dbExecute(duck_con, "CREATE VIEW experiment AS SELECT * FROM read_parquet('./sim_data/simulated_trials.parquet')")
experiment <- tbl(duck_con, "experiment")

experiment |>
  select(rep, phase, testgroup, trialtype, trialnumber) |>
  distinct() |>
  group_by(rep, phase, testgroup, trialtype) |>
  slice_max(order_by = trialnumber, n = 200) -> selected_trials

experiment |>
  group_by(rep, phase, testgroup, trialtype, trialnumber) |>
  mutate(t_min = min(t)) |>
  ungroup() |>
  mutate(trialnumber = if_else(t - t_min > 50, trialnumber + 1, trialnumber)) |>
  group_by(rep, phase, testgroup, trialtype, trialnumber) |>
  mutate(t_min = max(t_min)) |>
  ungroup() |>
  mutate(t_rel = t - t_min) |>
  select(-trialtype) -> experiment_t

selected_trials |>
  left_join(experiment_t) |>
  ungroup() -> experiment_extract

experiment_extract |>
  left_join(results |> filter(abs(gamma - 0.925) < 0.001, model_switch == 2)) |>
  filter(trialtype == "CSp", phase == 2, between(t_rel,-20,2)) |>
  collect() -> cc_raw_model_data

# dbExecute(duck_con, "CREATE VIEW results_bs AS SELECT * FROM read_parquet('./sim_data/belief_sim/*.parquet')")
bsresults <- tbl(duck_con, "results_bs")

experiment_extract |>
  left_join(bsresults |> filter(abs(discount_factor - 0.925) < 0.001)) |>
  filter(trialtype == "CSp", phase == 2, between(t_rel, -20, 6)) |>
  collect() -> bs_raw_model_data

#max rpe in model ~0.2, 90th percentile da signal in regression ~15, so scale 75x


cc_raw_model_data |>
  na.omit() |>
  group_by(rep, trialnumber, testgroup) |>
  reframe(value = mean(value[t_rel < 2]), rpe = 75*mean(rpe[t_rel == 2])) -> cc_sum_model_data

bs_raw_model_data |>
  na.omit() |> 
  group_by(rep, trialnumber, testgroup) |>
  reframe(value = mean(value[t_rel < 2]), rpe = 75*mean(rpe[t_rel == 2])) -> bs_sum_model_data

# fit value -> lick transform

panel5_data |>
  group_by(group) |> 
  slice_sample(n=500) |> 
  pull(pre_licks) |>
  quantile(probs = seq(0, 1, 0.01)) |>
  as_tibble() |>
  mutate(quant = row_number()) |>
  rename(licks = value) -> lick_quant

cc_sum_model_data |>
  pull(value) |>
  quantile(probs = seq(0, 1, 0.01)) |>
  as_tibble() |>
  mutate(quant = row_number()) |> 
  rename(cc_value = value) -> cc_quant

bs_sum_model_data |>
  pull(value) |>
  quantile(probs = seq(0, 1, 0.01)) |>
  as_tibble() |>
  mutate(quant = row_number()) |> 
  rename(bs_value = value)-> bs_quant

lick_quant |> left_join(cc_quant) |> left_join(bs_quant)  -> quants

glm(licks~cc_value,data=quants,family=poisson(link='log')) -> cc_model
glm(licks~bs_value,data=quants,family=poisson(link='log')) -> bs_model


cc_sum_model_data |>
  rename(cc_value = value) |> modelr::add_predictions(cc_model,var='licks') |> 
  mutate(licks = as.numeric(licks)) |> 
  group_by(testgroup,rep) |>
  nest() |>
  mutate(model = map(data, ~ lm(rpe ~ licks, data = .x))) |>
  mutate(tidy = map(model, broom::tidy)) |>
  unnest(tidy) |>
  filter(term == "licks") |>
  group_by(testgroup) |>
  reframe(mestimate = mean(estimate), std.error = sd(estimate)) |>
  ggplot() +
  aes(testgroup, mestimate, ymin = mestimate - std.error, ymax = mestimate + std.error) +
  geom_point() +
  geom_errorbar()+ylim(-3.2,3.3)+theme_cowplot()



bs_sum_model_data |>
  rename(bs_value = value) |> modelr::add_predictions(bs_model,var='licks') |> 
  rowwise() |> 
  mutate(licks = if_else(licks<0,rbinom(n=1,size=1,prob=0.2),licks)) |> 
  group_by(testgroup,rep) |>
  nest() |>
  mutate(model = map(data, ~ lm(rpe ~ licks, data = .x))) |>
  mutate(tidy = map(model, broom::tidy)) |>
  unnest(tidy) |>
  filter(term == "licks") |>
  na.omit() |> 
  group_by(testgroup) |>
  reframe(mestimate = mean(estimate), std.error = sd(estimate)) |>
  ggplot() +
  aes(testgroup, mestimate, ymin = mestimate - std.error, ymax = mestimate + std.error) +
  geom_point() +
  geom_errorbar()+ylim(-3.2,3.3)+theme_cowplot()