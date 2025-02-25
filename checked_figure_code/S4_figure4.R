sim_summary <- read_parquet("td_sim_summary_results.parquet")

#Fig S4a

sim_summary |>
  filter(trialtype == "CSp", t_rel == 2) |>
  group_by(rep, testgroup, model_switch,gamma) |>
  mutate(ref_rpe = mean(rpe[phase == 1])) |>
  filter(phase == 2) |>
  group_by(testgroup, model_switch,gamma) |>
  reframe(norm_rpe = mean(rpe / ref_rpe), sd_norm_rpe = sd(rpe / ref_rpe)) |>
  ggplot() +
  aes(gamma^5, norm_rpe, ymin = norm_rpe - sd_norm_rpe, ymax = norm_rpe + sd_norm_rpe, color = factor(testgroup)) +
  geom_point() +
  geom_line()+
  facet_grid(~model_switch) +
  theme_cowplot() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Normalized TD Error") +
  scale_y_continuous(breaks = c(0, 0.5, 1),limits = c(0,1.1))

#Fig S4b

sim_summary |>
  filter(trialtype == "CSp", t_rel == 2) |>
  group_by(rep, testgroup, model_switch,gamma) |>
  mutate(ref_rpe = mean(rpe[phase == 1])) |>
  filter(phase == 2) |>
  group_by(testgroup, model_switch,gamma) |>
  reframe(norm_rpe = mean(rpe / 1), sd_norm_rpe = sd(rpe / 1)) |>
  ggplot() +
  aes(gamma^5, norm_rpe, ymin = norm_rpe - sd_norm_rpe, ymax = norm_rpe + sd_norm_rpe, color = factor(testgroup)) +
  geom_point() +
  geom_line()+
  facet_grid(~model_switch) +
  theme_cowplot() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Normalized TD Error") +
  scale_y_continuous(limits = c(0,0.5))

# Fig S4d

dbExecute(duck_con, "CREATE VIEW alter_p AS SELECT * FROM read_parquet('./sim_data/alter_transition//*.parquet')")
alter_p <- tbl(duck_con, "alter_p")

experiment_extract |> left_join(alter_p) |> filter(trialtype == "CSp", t_rel == 2) |> collect() |> 
  group_by(rep, testgroup, trans_p) |>
  mutate(ref_rpe = mean(rpe[phase == 1])) |>
  filter(phase == 2) -> alter_p_data

alter_p_data |> 
  group_by(testgroup, trans_p) |>
  reframe(norm_rpe = mean(rpe / ref_rpe), sd_norm_rpe = sd(rpe / ref_rpe)) |> 
  ggplot() +
  aes(1-trans_p, norm_rpe, ymin = norm_rpe - sd_norm_rpe, ymax = norm_rpe + sd_norm_rpe, color = factor(testgroup)) +
  geom_line() +
  scale_x_continuous(transform = 'log10')+ylim(0,1.2)+theme_cowplot()

alter_p_data |> 
  group_by(testgroup, trans_p) |>
  reframe(norm_rpe = mean(rpe / 1), sd_norm_rpe = sd(rpe / 1)) |> 
  ggplot() +
  aes(1-trans_p, norm_rpe, ymin = norm_rpe - sd_norm_rpe, ymax = norm_rpe + sd_norm_rpe, color = factor(testgroup)) +
  geom_line() +
  scale_x_continuous(transform = 'log10')+ylim(0,0.5)+theme_cowplot()
  
