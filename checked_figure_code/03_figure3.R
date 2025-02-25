# Figure 3

# simulation code
# 1. generate simulated experiments in matlab
# 2. run simulations in python or matlab (belief-state) 

#model summarization

require(tidyverse)
require(arrow)
require(duckdb)

#model labels 1: CSC (no ITI), 2: Cue-Context, 3: CSC with ITI, 4: Belief State

duck_con <- dbConnect(duckdb::duckdb(dbdir = "./duck1.db"))
dbExecute(duck_con, "PRAGMA memory_limit='40GB'")

dbExecute(duck_con, "CREATE VIEW results AS SELECT * FROM read_parquet('./sim_data/m1_3_results/*.parquet')")
results <- tbl(duck_con, "results")

dbExecute(duck_con, "CREATE VIEW experiment AS SELECT * FROM read_parquet('./sim_data/simulated_trials.parquet')")
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
  left_join(results) |>
  group_by(rep, phase, testgroup, trialtype, model_switch, gamma, t_rel) |>
  summarise(value = mean(value), rpe = mean(rpe)) -> summary_results

summary_results |> collect() -> summary_results

dbExecute(duck_con, "CREATE VIEW results_bs AS SELECT * FROM read_parquet('./sim_data/belief_sim/*.parquet')")
bsresults <- tbl(duck_con, "results_bs")

experiment_extract |>
  left_join(bsresults) |>
  group_by(rep, phase, testgroup, trialtype, discount_factor, t_rel) |>
  summarise(value = mean(value), rpe = mean(rpe)) -> summary_bs_results

summary_bs_results |> collect() -> summary_bs_results

summary_bs_results |>
  ungroup() |>
  mutate(model_switch = 4) |>
  rename(gamma = discount_factor) |>
  bind_rows(ungroup(summary_results)) -> summary_results

summary_results |> write_parquet("td_sim_summary_results.parquet")

# figure

sim_summary <- read_parquet("td_sim_summary_results.parquet")

# fig 3d

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSp", testgroup < 3) |>
  group_by(trialtype, testgroup, model_switch, t_rel) |>
  reframe(value = mean(value), rpe = mean(rpe)) |>
  ggplot() +
  aes(t_rel, value, color = factor(testgroup)) +
  facet_grid(trialtype ~ model_switch) +
  geom_line() +
  xlim(-5, 30)

# fig 3e

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSp", testgroup < 3) |>
  group_by(trialtype, testgroup, model_switch, t_rel) |>
  reframe(value = mean(value), rpe = mean(rpe)) |>
  ggplot() +
  aes(t_rel, rpe, color = factor(testgroup)) +
  facet_grid(trialtype ~ model_switch) +
  geom_point() +
  xlim(-5, 30)

# fig 3f

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, trialtype == "CSp", t_rel == 2) |>
  group_by(rep, testgroup, model_switch) |>
  mutate(ref_rpe = mean(rpe[phase == 1])) |>
  filter(phase == 2) |>
  group_by(testgroup, model_switch) |>
  reframe(norm_rpe = mean(rpe / ref_rpe), sd_norm_rpe = sd(rpe / ref_rpe)) |>
  ggplot() +
  aes(testgroup, norm_rpe, ymin = norm_rpe - sd_norm_rpe, ymax = norm_rpe + sd_norm_rpe, fill = factor(testgroup)) +
  geom_col() +
  geom_errorbar(width = 0.33) +
  facet_grid(~model_switch) +
  theme_cowplot() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Normalized TD Error") +
  scale_y_continuous(breaks = c(0, 0.5, 1))