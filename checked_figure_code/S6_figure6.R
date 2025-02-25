# panel C

experiment_extract |>
  left_join(bsresults) |>
  filter(abs(discount_factor - 0.925) < 0.001, t_rel == 19, trialtype == "CSp") |>
  collect() -> reward_sim

reward_sim |>
  mutate(rewarded = if_else(rpe > 0, "rewarded", "omission")) |>
  group_by(rep, testgroup, rewarded) |>
  reframe(mrpe = mean(rpe)) |>
  group_by(testgroup, rewarded) |>
  reframe(mean_rpe = mean(mrpe), sd_rpe = sd(mrpe)) -> rewarded_sim_sum

extinct_results |>
  filter(phase == 2, trialtype == "CSp", t_rel == 19, between(trialnumber, 750, 1000)) |>
  group_by(rep, testgroup) |>
  reframe(mrpe = mean(rpe)) |>
  group_by(testgroup) |>
  reframe(mean_rpe = mean(mrpe), sd_rpe = sd(mrpe)) |>
  mutate(rewarded = "omission") -> rewarded_extinction


bind_rows(rewarded_sim_sum, rewarded_extinction) |> ggplot() +
  aes(testgroup, mean_rpe, fill = testgroup, ymin = mean_rpe - sd_rpe, ymax = mean_rpe + sd_rpe) +
  geom_col() +
  geom_errorbar(width = 0.2) +
  facet_grid(rewarded ~ ., scales = "free_y") +
  theme_cowplot() +
  theme(legend.position = "none")

# Panel B (upper)

labelled_photo_data |>
  filter(trialtype == "go_omit") |>
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
  aes((time - 40) / 20, mean_da, colour = factor(panel)) +
  geom_line()

labelled_photo_data |>
  filter(trialtype == "go_omit") |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  filter(between(time, 110, 140)) |>
  group_by(mouse, day, panel, trialnumber) |>
  reframe(mean_da = sum(data)) -> mouse_trial_data

mouse_trial_data |>
  group_by(mouse, day, panel) |>
  reframe(mda = mean(mean_da)) -> panel_d_data

panel_d_data |>
  filter(panel != "A") |>
  group_by(day, panel) |>
  reframe(mm = mean(mda), sem = sd(mda) / sqrt(n())) |>
  ggplot() +
  aes(day, mm, ymin = mm - sem, ymax = mm + sem, color = panel) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0)

mouse_trial_data |>
  group_by(panel) |>
  filter(day == max(day) | day == min(day)) |>
  mutate(last_day = day == max(day)) -> mouse_trial_stats_data

mouse_trial_stats_data |> filter(panel == "B") -> mouse_trial_stats_b
lmer(mean_da ~ last_day + (1 | mouse), data = mouse_trial_stats_b, REML = F) -> day_by_day_model_b
pairs(emmeans(day_by_day_model_b, ~last_day)) |> summary()

mouse_trial_stats_data |> filter(panel == "C") -> mouse_trial_stats_c
lmer(mean_da ~ last_day + (1 | mouse), data = mouse_trial_stats_c, REML = F) -> day_by_day_model_c
pairs(emmeans(day_by_day_model_c, ~last_day)) |> summary()

mouse_trial_stats_data |> filter(panel == "D") -> mouse_trial_stats_d
lmer(mean_da ~ last_day + (1 | mouse), data = mouse_trial_stats_d, REML = F) -> day_by_day_model_d
pairs(emmeans(day_by_day_model_d, ~last_day)) |> summary()

# panel d (left)

panel_d_data |>
  group_by(panel) |>
  filter(day == max(day)) -> panel_d_points

panel_d_points |>
  group_by(panel) |>
  reframe(mean_da = mean(mda), sem = sd(mda) / sqrt(n())) |>
  ggplot() +
  aes(panel, mean_da, fill = panel) +
  geom_col() +
  geom_errorbar(width = 0.2, aes(ymin = mean_da - sem, ymax = mean_da + sem)) +
  geom_point(data = panel_d_points, aes(y = mda), shape = 1)

mouse_trial_data |>
  group_by(panel) |>
  filter(day == max(day)) -> mouse_stats_data

lmer(mean_da ~ panel + (1 | mouse), data = mouse_stats_data, REML = F) -> mouse_stats_data_model
pairs(emmeans(mouse_stats_data_model, ~panel)) |> summary()

# panel b (lower)


labelled_photo_data |>
  filter(trialtype == "go") |>
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
  aes((time - 40) / 20, mean_da, colour = factor(panel)) +
  geom_line()

labelled_photo_data |>
  filter(trialtype == "go") |>
  mutate(panel = case_when(
    day < 5 ~ "A",
    between(day, 5, 9) & group == "degradation" ~ "B",
    between(day, 5, 9) & group == "cuedrew" ~ "C",
    between(day, 13, 15) ~ "D",
    .default = "E"
  )) |>
  filter(panel != "E") |>
  filter(between(time, 110, 140)) |>
  group_by(mouse, day, panel, trialnumber) |>
  reframe(mean_da = max(data)) -> mouse_trial_data

mouse_trial_data |>
  group_by(mouse, day, panel) |>
  reframe(mda = mean(mean_da)) -> panel_d_data

panel_d_data |>
  filter(panel != "A") |>
  group_by(day, panel) |>
  reframe(mm = mean(mda), sem = sd(mda) / sqrt(n())) |>
  ggplot() +
  aes(day, mm, ymin = mm - sem, ymax = mm + sem, color = panel) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0)

mouse_trial_data |>
  group_by(panel) |>
  filter(day == max(day) | day == min(day)) |>
  mutate(last_day = day == max(day)) -> mouse_trial_stats_data

mouse_trial_stats_data |> filter(panel == "B") -> mouse_trial_stats_b
lmer(mean_da ~ last_day + (1 | mouse), data = mouse_trial_stats_b, REML = F) -> day_by_day_model_b
pairs(emmeans(day_by_day_model_b, ~last_day)) |> summary()

mouse_trial_stats_data |> filter(panel == "C") -> mouse_trial_stats_c
lmer(mean_da ~ last_day + (1 | mouse), data = mouse_trial_stats_c, REML = F) -> day_by_day_model_c
pairs(emmeans(day_by_day_model_c, ~last_day)) |> summary()

# panel d (right)

panel_d_data |>
  group_by(panel) |>
  filter(day == max(day)) -> panel_d_points

panel_d_points |>
  group_by(panel) |>
  reframe(mean_da = mean(mda), sem = sd(mda) / sqrt(n())) |>
  ggplot() +
  aes(panel, mean_da, fill = panel) +
  geom_col() +
  geom_errorbar(width = 0.2, aes(ymin = mean_da - sem, ymax = mean_da + sem)) +
  geom_point(data = panel_d_points, aes(y = mda), shape = 1)

mouse_trial_data |>
  group_by(panel) |>
  filter(day == max(day)) -> mouse_stats_data

lmer(mean_da ~ panel + (1 | mouse), data = mouse_stats_data, REML = F) -> mouse_stats_data_model
pairs(emmeans(mouse_stats_data_model, ~panel)) |> summary()

# panel f

labelled_photo_data |>
  filter(trialtype == "unpred_water") |>
  filter(lick_latency < 1) |>
  mutate(time = round(time - lick_latency * 20, 0)) |>
  group_by(time, mouse, day) |>
  reframe(mda = mean(data)) |>
  group_by(time, day) |>
  reframe(mda = mean(mda)) |>
  ggplot() +
  aes(time, mda, color = factor(day)) +
  geom_line() +
  xlim(80, 150)

labelled_photo_data |>
  filter(trialtype == "unpred_water") |>
  filter(lick_latency < 1) |>
  mutate(time = round(time - lick_latency * 20, 0)) |>
  filter(between(time, 80, 150)) |>
  group_by(mouse, day, trialnumber) |>
  reframe(maxda = max(data)) |>
  group_by(mouse, day) |>
  reframe(mda = mean(maxda)) -> panel_f_stats_data

panel_f_stats_data |>
  group_by(day) |>
  reframe(meanda = mean(mda), sem = sd(mda) / sqrt(n())) |>
  ggplot() +
  aes(day, meanda) +
  geom_point() +
  geom_smooth(method = "lm", se = F) +
  geom_errorbar(width = 0, aes(ymin = meanda - sem, ymax = meanda + sem))+scale_y_continuous(breaks = c(8,10),limits = c(6.5,12))+theme_cowplot()

lm(mda ~ I(day - 4), data = panel_f_stats_data) |> broom::tidy()

#
experiment |>
  select(rep, phase, testgroup, trialtype, trialnumber) |>
  distinct() |>
  group_by(rep, phase, testgroup, trialtype) -> all_trials

all_trials |>
  left_join(experiment_t) |>
  ungroup() -> all_trials_extract

all_trials_extract |>
  left_join(bsresults) |>
  filter(abs(discount_factor - 0.925) < 0.001, trialtype == "degrade", t_rel == 20) |>
  collect() -> all_bs_degrade

all_bs_degrade |>
  mutate(day = case_when(
    trialnumber < 500 ~ 1,
    trialnumber > 3500 ~ 5,
    .default = 0
  )) |>
  filter(day > 0) |>
  group_by(rep, day) |>
  reframe(mrpe = mean(rpe)) |> 
  group_by(day) |> 
  reframe(mm = mean(mrpe),sdm = sd(mrpe)) |> 
  ggplot()+aes(factor(day),mm,ymin=mm-sdm,ymax=mm+sdm)+
  geom_col()+
  geom_errorbar(width=0.2)+scale_y_continuous(limits=c(0,1.2),breaks=c(0,1))+theme_cowplot()
