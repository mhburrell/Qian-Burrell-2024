# Panel B
require(lmerTest)
require(emmeans)
labelled_photo_data |>
  filter(trialtype %in% c("go", "go_omit"), all_licks > 0) |>
  filter(between(time, 30, 70)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  group_by(mouse, time, day, panel) |>
  reframe(mean_da = mean(data)) |>
  group_by(time, day, panel) |>
  reframe(mean_da = mean(mean_da)) |>
  ggplot() +
  aes((time - 40) / 20, mean_da, colour = factor(day)) +
  facet_grid(panel ~ .) +
  geom_line()


labelled_photo_data |>
  filter(trialtype %in% c("go", "go_omit")) |>
  filter(between(time, 30, 70)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  group_by(trial, mouse, day, panel) |>
  reframe(max_da = max(data)) -> panel_b_points

panel_b_points |>
  group_by(day, panel) |>
  reframe(mean_da = mean(max_da), sem_da = sd(max_da) / sqrt(n())) |>
  ggplot() +
  aes(day, mean_da, colour = panel, ymax = mean_da + sem_da, ymin = mean_da - sem_da) +
  geom_point() +
  geom_errorbar() +
  geom_line() +
  facet_grid(. ~ panel, scales = "free_x") +
  ylim(0, 16)

panel_b_points |> group_by(panel) |> 
  filter(day == max(day)|day==min(day)) |>
  mutate(last_day = day == max(day)) -> panel_b_stats_data

panel_b_stats_data |> filter(panel=='A') -> panel_b_stats_a
lmer(max_da ~ last_day + (1 | mouse), data = panel_b_stats_a, REML = F) -> panel_b_model_a 
pairs(emmeans(panel_b_model_a, ~last_day)) |> summary()

panel_b_stats_data |> filter(panel=='B') -> panel_b_stats_b
lmer(max_da ~ last_day + (1 | mouse), data = panel_b_stats_b, REML = F) -> panel_b_model_b
pairs(emmeans(panel_b_model_b, ~last_day)) |> summary()

panel_b_stats_data |> filter(panel=='C') -> panel_b_stats_c
lmer(max_da ~ last_day + (1 | mouse), data = panel_b_stats_c, REML = F) -> panel_b_model_c
pairs(emmeans(panel_b_model_c, ~last_day)) |> summary()

panel_b_stats_data |> filter(panel=='D') -> panel_b_stats_d
lmer(max_da ~ last_day + (1 | mouse), data = panel_b_stats_d, REML = F) -> panel_b_model_d
pairs(emmeans(panel_b_model_d, ~last_day)) |> summary()

# Panel D

labelled_photo_data |>
  filter(trialtype %in% c("go", "go_omit")) |>
  filter(between(time, 40, 70)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  filter(all_licks > 0 | panel == "D") |>
  group_by(trial, mouse, day, panel) |>
  reframe(max_da = max(data)) |>
  group_by(panel) |>
  filter(day == max(day)) -> panel_d_data

panel_d_data |> 
  ungroup() |>
  group_by(mouse, panel) |>
  reframe(mean_da = mean(max_da)) -> panel_d_points

panel_d_points |>
  group_by(panel) |>
  reframe(meanda = mean(mean_da), semda = sd(mean_da) / sqrt(n())) |>
  ggplot() +
  aes(panel, meanda, color = panel, fill = panel) +
  geom_col() +
  geom_errorbar(width = 0.2, aes(ymax = meanda + semda, ymin = meanda - semda)) +
  geom_point(data = panel_d_points, aes(panel, mean_da), shape = 1, color='black') +
  theme_cowplot() +
  theme(length.position = "none")

lmer(max_da ~ panel + (1 | mouse), data = panel_d_data, REML = F) -> panel_d_model
pairs(emmeans(panel_d_model, ~panel)) |> summary()

#Model Data

sim_summary |> filter(abs(gamma-0.925)<0.001,t_rel==2,trialtype=='CSp',phase==2,model_switch==4) |> group_by(rep,testgroup) |> 
  reframe(mean_rpe = mean(rpe)) |> 
  group_by(testgroup) |> 
  reframe(mrpe = mean(mean_rpe),sdrpe = sd(mean_rpe)) -> odor_a



#Extinction data

dbExecute(duck_con, "CREATE VIEW ext_results AS SELECT * FROM read_parquet('./sim_data/bs_ext/*.parquet')")
ext_results <- tbl(duck_con,'ext_results')

dbExecute(duck_con, "CREATE VIEW ext_experiment AS SELECT * FROM read_parquet('./sim_data/simulated_extinction_trials.parquet')")
ext_experiment <- tbl(duck_con, "ext_experiment")

ext_experiment |>
  select(rep, phase, testgroup, trialtype, trialnumber) |>
  distinct() |>
  group_by(rep, phase, testgroup, trialtype)  -> ext_selected_trials

ext_experiment |>
  group_by(rep, phase, testgroup, trialtype, trialnumber) |>
  mutate(t_min = min(t)) |>
  ungroup() |>
  mutate(trialnumber = if_else(t - t_min > 50, trialnumber + 1, trialnumber)) |>
  group_by(rep, phase, testgroup, trialtype, trialnumber) |>
  mutate(t_min = max(t_min)) |>
  ungroup() |>
  mutate(t_rel = t - t_min) |>
  select(-trialtype) -> ext_experiment_t

ext_selected_trials |>
  left_join(ext_experiment_t) |>
  ungroup() -> ext_experiment_extract

ext_experiment_extract |> 
  left_join(ext_results) |> 
  ungroup() |> collect() -> extinct_results

#day 3 extinction

extinct_results |> filter(phase==2,trialtype=='CSp',t_rel==2,between(trialnumber,750,1000)) |> group_by(rep,testgroup) |> 
  reframe(mean_rpe = mean(rpe)) |> 
  group_by(testgroup) |> 
  reframe(mrpe = mean(mean_rpe),sdrpe = sd(mean_rpe)) -> odor_a_extinction
  
bind_rows(odor_a,odor_a_extinction) |> 
  ggplot()+aes(testgroup,mrpe,ymin=mrpe-sdrpe,ymax=mrpe+sdrpe,fill=factor(testgroup))+
  geom_col()+
  geom_errorbar(width=0.2)

# Panel E

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSm", model_switch %in% c(4)) |>
  group_by(testgroup, t_rel) |>
  reframe(value = mean(value), rpe = mean(rpe)) |>
  ggplot() +
  aes(t_rel, value, color = factor(testgroup)) +
  geom_line() +
  xlim(-10, 40)

sim_summary |>
  filter(abs(gamma - 0.925) < 0.001, phase == 2, trialtype == "CSm", model_switch %in% c(4), t_rel == 2) |>
  group_by(testgroup) |>
  reframe(mean_rpe = mean(rpe), sd_rpe = sd(rpe)) -> odor_b

extinct_results |> filter(phase==2,trialtype=='CSm',t_rel==2,between(trialnumber,750,1000)) |> group_by(rep,testgroup) |> 
  reframe(mrpe = mean(rpe)) |> 
  group_by(testgroup) |> 
  reframe(mean_rpe = mean(mrpe),sd_rpe = sd(mrpe)) -> odor_b_extinction

bind_rows(odor_b,odor_b_extinction) |> 
  ggplot() +
  aes(testgroup, mean_rpe, fill = testgroup, ymin = mean_rpe - sd_rpe, ymax = mean_rpe + sd_rpe) +
  geom_col() +
  geom_errorbar(width = 0.2) +
  theme_cowplot() +
  theme(legend.position = "none")

# Panel F

labelled_photo_data |>
  filter(trialtype %in% c("no_go")) |>
  filter(between(time, 30, 70)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  group_by(panel) |>
  filter(day == max(day)) |>
  group_by(mouse, time, day, panel) |>
  reframe(mean_da = mean(data)) |>
  group_by(time, day, panel) |>
  reframe(mean_da = mean(mean_da)) |>
  ggplot() +
  aes((time - 40) / 20, mean_da, colour = factor(day)) +
  facet_grid(panel ~ .) +
  geom_line()


labelled_photo_data |>
  filter(trialtype %in% c("no_go")) |>
  filter(between(time, 45, 60)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E", panel != "A") |>
  group_by(trial, mouse, day, panel) |>
  reframe(max_da = sum(data)) -> day_by_day

day_by_day |> 
  group_by(day, panel) |>
  reframe(mean_da = mean(max_da), sem_da = sd(max_da) / sqrt(n())) |>
  ggplot() +
  aes(day, mean_da, colour = panel, ymax = mean_da + sem_da, ymin = mean_da - sem_da) +
  geom_point() +
  geom_errorbar() +
  geom_line() +
  facet_grid(. ~ panel, scales = "free_x")

day_by_day |> group_by(panel) |> 
  filter(day == max(day)|day==min(day)) |>
  mutate(last_day = day == max(day)) -> day_by_day_stats_data

day_by_day_stats_data |> filter(panel=='B') -> day_by_day_stats_b
lmer(max_da ~ last_day + (1 | mouse), data = day_by_day_stats_b, REML = F) -> day_by_day_model_b
pairs(emmeans(day_by_day_model_b, ~last_day)) |> summary()

day_by_day_stats_data |> filter(panel=='C') -> day_by_day_stats_c
lmer(max_da ~ last_day + (1 | mouse), data = day_by_day_stats_c, REML = F) -> day_by_day_model_c
pairs(emmeans(day_by_day_model_c, ~last_day)) |> summary()

day_by_day_stats_data |> filter(panel=='D') -> day_by_day_stats_d
lmer(max_da ~ last_day + (1 | mouse), data = day_by_day_stats_d, REML = F) -> day_by_day_model_d
pairs(emmeans(day_by_day_model_d, ~last_day)) |> summary()

labelled_photo_data |>
  filter(trialtype %in% c("no_go")) |>
  filter(between(time, 45, 60)) |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  group_by(trial, mouse, day, panel) |>
  reframe(max_da = sum(data)) |>
  group_by(panel) |>
  filter(day == max(day)) |>
  ungroup()  -> panel_f_data

panel_f_data |>
  group_by(mouse, panel) |>
  reframe(mean_da = mean(max_da)) -> panel_f_points

panel_f_points |>
  group_by(panel) |>
  reframe(meanda = mean(mean_da), semda = sd(mean_da) / sqrt(n())) |>
  ggplot() +
  aes(panel, meanda, color = panel, fill = panel) +
  geom_col() +
  geom_errorbar(width = 0.2, aes(ymax = meanda + semda, ymin = meanda - semda)) +
  geom_point(data = panel_f_points, aes(panel, mean_da), shape = 1) +
  theme_cowplot() +
  theme(length.position = "none")

lmer(max_da ~ panel + (1 | mouse), data = panel_f_data, REML = F) -> panel_f_model
pairs(emmeans(panel_f_model, ~panel))
