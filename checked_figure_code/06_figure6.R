# Figure 6
require(tidyverse)
require(arrow)
require(duckdb)
require(cowplot)

source("06_figure6_helperfunctions.R")
# source("06_figure6_dataprep.R") run once

rnn_data <- open_dataset("./sim_data/labelled_rnn_data/", unify_schemas = TRUE)

rnn_data |>
  to_duckdb() |>
  filter(trained == TRUE, hidden_size == 50, test == FALSE) |>
  group_by(seed, pca_group, condition, aligned_time, t_type) |>
  summarise(value = mean(value), rpe = mean(rpe), V1 = mean(V1), V2 = mean(V2), V3 = mean(V3)) |>
  collect() -> rnn_50_data

# Figure 6c

rnn_50_data |>
  filter(aligned_time %in% c(-1, 7, 8)) |>
  mutate(event = case_when(
    aligned_time == -1 & t_type == "Cue A Rewarded" ~ "Cue A",
    aligned_time == 7 & t_type == "Cue A Rewarded" ~ "Predicted Reward",
    aligned_time == 7 & t_type == "Cue A Unrewarded" ~ "Omission",
    aligned_time == -1 & t_type == "Cue B" ~ "Cue B",
    aligned_time == -1 & t_type == "Cue C Rewarded" ~ "Cue C",
    aligned_time == 8 & t_type == "Degradation" ~ "Unpredicted Reward",
    .default = NA
  )) |>
  filter(!is.na(event)) |>
  group_by(condition, event) |>
  summarise(m_rpe = mean(rpe), sd_rpe = sd(rpe)) |>
  ggplot() +
  aes(condition, m_rpe, ymin = m_rpe - sd_rpe, ymax = m_rpe + sd_rpe) +
  geom_col() +
  geom_errorbar(width = 0.5) +
  facet_wrap(~event) +
  theme_cowplot()

# Fig 6d

rnn_data |>
  to_duckdb() |>
  filter(seed == 175, pca_group == "pretrained", test == FALSE, trained == TRUE) |>
  collect() |>
  shift_iti() |>
  group_by(aligned_time, t_type, condition) |>
  summarise(rpe = mean(rpe), value = mean(value), V1 = mean(V1), V2 = mean(V2), V3 = mean(V3)) -> rnn_50_example

rnn_50_example |>
  select(aligned_time, t_type, condition, rpe, value) |>
  filter(t_type != "Blank") |>
  pivot_longer(cols = c(rpe, value)) |>
  ggplot() +
  aes(aligned_time, value, color = condition) +
  geom_line() +
  facet_grid(t_type ~ name) +
  theme_cowplot() +
  xlim(-5, 20) +
  theme(legend.position = "bottom")

rnn_data |>
  select(hidden_size) |>
  distinct() |>
  collect() -> hidden_size_list


for (i in hidden_size_list$hidden_size) {
  rnn_data |>
    filter(test == FALSE, trained == TRUE, hidden_size == i, pca_group == "pretrained") |>
    collect() |>
    select_if(~ sum(!is.na(.)) > 0) |>
    group_by(hidden_size) |>
    write_dataset(path = "./for_belief_r2")
}

rnn_50_example |>
  group_by(condition) |>
  mutate(tp = case_when(
    aligned_time == 0 ~ "c",
    aligned_time == 8 ~ "r",
    .default = "n"
  )) |>
  ungroup() |>
  arrange(aligned_time) |>
  write_parquet(sink = "3dplotdata.parquet")

rnn_data |>
  to_duckdb() |>
  filter(seed == 175, pca_group == "pretrained", test == TRUE, trained == TRUE) |>
  collect() |>
  shift_iti() |>
  group_by(aligned_time, t_type, condition) |>
  summarise(rpe = mean(rpe), value = mean(value), V1 = mean(V1), V2 = mean(V2), V3 = mean(V3)) |>
  group_by(condition) |>
  mutate(tp = case_when(
    aligned_time == 0 ~ "c",
    aligned_time == 8 ~ "r",
    .default = "n"
  )) |>
  ungroup() |>
  arrange(aligned_time) |>
  write_parquet(sink = "3dplotdata_test.parquet")

# Fig 6e

#run analysis in MATLAB

r2_table <- read_parquet("r2_table.parquet")

rnn_data |>
  to_duckdb() |>
  select(seed, hidden_size) |>
  distinct() |>
  collect() -> seed_hidden_size

r2_table |>
  left_join(seed_hidden_size) |>
  group_by(hidden_size, condition) |>
  reframe(mean_r2 = mean(r2), sd_r2 = sd(r2)) |>
  ggplot() +
  aes(hidden_size, mean_r2, color = condition) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = mean_r2 - sd_r2, ymax = mean_r2 + sd_r2), width = 0.5) +
  theme_cowplot() +
  scale_x_continuous(limits=c(0,52),breaks=seq(0,50,by=10)) +
  ylim(0, 1)

r2_table |>
  left_join(seed_hidden_size) |>
  group_by(hidden_size, condition) |>
  reframe(mean_r2 = mean(r2_9), sd_r2 = sd(r2_9)) |>
  ggplot() +
  aes(hidden_size, mean_r2, color = condition) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = mean_r2 - sd_r2, ymax = mean_r2 + sd_r2), width = 0.5) +
  theme_cowplot() +
  scale_x_continuous(limits=c(0,52),breaks=seq(0,50,by=10)) +
  ylim(0, 1)
