ms_summary <- read_parquet("./sim_data/ms_summary.parquet")

ms |>
  filter(trialtype == "CSp", rew == 1, t_trial == 1) |>
  select(rep, testgroup, sigma, tau, n_stimuli, rpe) |>
  pivot_wider(names_from = "testgroup", values_from = "rpe", names_prefix = "g") -> ms_cue_responses

ms |>
  filter(trialtype == "CSp", rew == 1, t_trial == 19, testgroup == 1) |>
  select(rep, testgroup, sigma, tau, n_stimuli, rpe) -> ms_rew_responses

left_join(ms_cue_responses, ms_rew_responses) |>
  group_by(sigma, tau, n_stimuli) |>
  na.omit() |>
  reframe(rpe = mean(rpe), g1 = mean(g1), g2 = mean(g2), g3 = mean(g3)) -> ms_summary

ms_responses |>
  filter(sigma == 0.15, tau == 0.99, n_stimuli == 25) |>
  select(g1, g2, g3, rpe) |>
  mutate(g2 = g2 / g1, g3 = g3 / g1) |>
  mutate(g1 = 1) |>
  pivot_longer(cols = c(g1, g2, g3, rpe), names_to = "group", values_to = "value") |>
  group_by(group) |>
  reframe(meanRPE = mean(value), sdRPE = sd(value)) -> best_ms_g2

ms_responses |>
  filter(sigma == 0.05, tau == 0.9, n_stimuli == 100) |>
  select(g1, g2, g3, rpe) |>
  mutate(g2 = g2 / g1, g3 = g3 / g1) |>
  mutate(g1 = 1) |>
  pivot_longer(cols = c(g1, g2, g3, rpe), names_to = "group", values_to = "value") |>
  group_by(group) |>
  reframe(meanRPE = mean(value), sdRPE = sd(value)) -> best_ms_rpe

best_ms_g2 |>
  filter(group == "rpe") |>
  ggplot() +
  aes(1, meanRPE) +
  geom_col() +
  ylim(0, 1) +
  theme_cowplot() -> p1

best_ms_rpe |>
  filter(group == "rpe") |>
  ggplot() +
  aes(1, meanRPE) +
  geom_col() +
  ylim(0, 1) +
  theme_cowplot() -> p2

best_ms_g2 |>
  filter(group != "rpe") |>
  ggplot() +
  aes(group, meanRPE) +
  geom_col() +
  theme_cowplot() -> p3

best_ms_rpe |>
  filter(group != "rpe") |>
  ggplot() +
  aes(group, meanRPE) +
  geom_col() +
  theme_cowplot() -> p4

plot_grid(p3, p1, p4, p2, nrow = 2, ncol = 2, rel_widths = c(2.618, 1)) -> p1_4


require(fastFMM)
labelled_photo_data |>
  filter(day %in% c(0, 4), between(time, 100, 150), trialtype == "go",lick_latency<0.2) |>
  pivot_wider(names_from = time, values_from = data, names_prefix = "y", names_sort = TRUE) -> data_for_fui

pred_rew_fui <- fui(y~factor(day)+(1|mouse),data=as.data.frame(data_for_fui),subj_ID = 'mouse')

flm_effect_plotter(pred_rew_fui,sigdisp = FALSE)+theme(legend.position = 'none') -> p5


labelled_photo_data |>
  filter(day %in% c(0, 4), between(time, 100, 150), trialtype == "go",lick_latency<0.2) |> 
  group_by(mouse,day,trialnumber) |> 
  reframe(max_sig = max(data)) |> 
  group_by(mouse,day) |> 
  reframe(mean_sig = mean(max_sig),n=n()) |> 
  filter(n>5) |> 
  select(-n) |> 
  pivot_wider(names_from = day, values_from = mean_sig,names_prefix = 'day') |> 
  mutate(day4_norm = day4/day0) |> na.omit() -> panel_k


