# Figure S3
require(fastFMM)
require(cowplot)

labelled_photo_data |>
  filter((lick_latency < 0.2 | between(lick_latency, 0.4, 0.8)), trialtype == "go", day > 2, between(time, 100, 180)) |>
  mutate(lick_fast = lick_latency < 0.3) |>
  pivot_wider(names_from = time, values_from = data, names_prefix = "y", names_sort = TRUE) |>
  select(starts_with("y"), mouse, lick_fast) -> pred_rew_data_wide

fui(y ~ lick_fast + (1 | mouse), data = as.data.frame(pred_rew_data_wide), subj_ID = "mouse") -> pred_rew_fit

# figS3a
source("flm_effect_plotter.R")
flm_effect_plotter(pred_rew_fit)


# example plots (b)
labelled_photo_data |>
  filter(trialnumber %in% c(0, 1, 2), trialtype %in% c("go"), day %in% c(3), mouse == "FgDA_04") |>
  ggplot() +
  aes(time, data, group = trialnumber) +
  geom_line() +
  geom_vline(xintercept = 110) +
  geom_vline(aes(xintercept = lick_latency * 20 + 109)) +
  facet_wrap(~trialnumber, ncol = 3) +
  xlim(100, 140) +
  theme_cowplot(font_size = 10) +
  theme(strip.background = element_blank(), strip.text = element_blank())

# Fig 3c

labelled_photo_data |>
  filter((lick_latency < 0.2 | between(day, 13, 15)), group == "degradation", trialtype == "go" | trialtype == "go_omit") |>
  group_by(mouse, day, trial, trialtype) |>
  reframe(odora = max(data[between(time, 40, 60)]), predrew = max(data[between(time, 110, 130)])) |>
  pivot_longer(cols = c("odora", "predrew")) |>
  filter(!(name == "predrew" & trialtype == "go_omit")) |>
  group_by(mouse, day, name) |>
  reframe(mean_data = mean(value)) |>
  group_by(day, name) |>
  reframe(mean_da = mean(mean_data), sem_da = sd(mean_data) / sqrt(n())) |>
  ggplot() +
  aes(day, mean_da, ymin = mean_da - sem_da, ymax = mean_da + sem_da, color = name) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot() +
  ylim(0, 14)

# Fig 3d

labelled_photo_data |>
  filter((lick_latency < 0.2 | between(lick_latency, 0.4, 0.8)), trialtype == "unpred_water", day > 2, between(time, 100, 180)) |>
  mutate(lick_fast = lick_latency < 0.3) |>
  pivot_wider(names_from = time, values_from = data, names_prefix = "y", names_sort = TRUE) |>
  select(starts_with("y"), mouse, lick_fast) -> unpred_rew_data_wide

fui(y ~ lick_fast + (1 | mouse), data = as.data.frame(unpred_rew_data_wide), subj_ID = "mouse") -> unpred_rew_fit

# source('flm_effect_plotter.R')
flm_effect_plotter(unpred_rew_fit)

#Fig 3e
labelled_photo_data |> 
  filter(trialtype == "unpred_water") |>
  filter(mouse == "FgDA_06", day == 7, trialnumber %in% c(25, 75, 34)) |>
  ggplot() +
  aes(time, data) +
  geom_line() +
  facet_wrap(~trialnumber) +
  xlim(100, 140) +
  geom_vline(aes(xintercept = lick_latency * 20 + 110)) +
  theme_cowplot(font_size=10) + theme(strip.background = element_blank(), strip.text = element_blank()) -> example_traces

# Fig S3f

labelled_photo_data |>
  filter(day < 10, lick_latency < 0.2, trialtype %in% c("go", "unpred_water", "c_reward"), between(time, 110, 120)) |>
  group_by(group, mouse, day, trialnumber, trialtype, lick_latency) |>
  reframe(maxDA = max(data)) -> rew_data

rew_data |>
  group_by(group, mouse, day, trialtype) |>
  reframe(meanDA = mean(maxDA)) |>
  group_by(group, day, trialtype) |>
  reframe(avgDA = mean(meanDA), semDA = sd(meanDA) / sqrt(n())) |>
  ggplot() +
  aes(day, avgDA, colour = trialtype) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = avgDA - semDA, ymax = avgDA + semDA), width = 0.1) +
  theme_cowplot(font_size = 10) +
  scale_x_continuous(limits = c(-0.1, 9.5), breaks = seq(0, 9, 1)) +
  scale_y_continuous(limits = c(0, 13)) +
  theme(legend.position = "none", strip.background = element_blank(), strip.text = element_blank()) +
  facet_grid(~group)

#Fig S3g

labelled_photo_data |> 
  filter(trialtype=='no_go',day<10|between(day,13,15),between(time,40,45)) |> 
  mutate(condition = case_when(
    day<5~'cond',
    day>10~'ext',
    .default = group
  )) |> 
  group_by(day,mouse,trial,condition) |> 
  reframe(max_sig = max(data)) |> 
  group_by(day,condition,mouse) |> 
  reframe(mean_sig = mean(max_sig)) |> 
  group_by(day,condition) |> 
  reframe(mda = mean(mean_sig),semda = sd(mean_sig)/sqrt(n())) |> 
  ggplot() + aes(day,mda,color=condition,ymin = mda-semda,ymax = mda+semda) +
  geom_point() +
  geom_line() +
  geom_errorbar(width=0) + theme_cowplot()

#Fig 3h

labelled_photo_data |> 
  filter(trialtype=='no_go',day<10|between(day,13,15),between(time,45,60)) |> 
  mutate(condition = case_when(
    day<5~'cond',
    day>10~'ext',
    .default = group
  )) |> 
  group_by(day,mouse,trial,condition) |> 
  reframe(max_sig = sum(data)) |> 
  group_by(day,condition,mouse) |> 
  reframe(mean_sig = mean(max_sig)) |> 
  group_by(day,condition) |> 
  reframe(mda = mean(mean_sig),semda = sd(mean_sig)/sqrt(n())) |> 
  ggplot() + aes(day,mda,color=condition,ymin = mda-semda,ymax = mda+semda) +
  geom_point() +
  geom_line() +
  geom_errorbar(width=0) + theme_cowplot()

#Fig 3i

labelled_photo_data |> 
  filter(trialtype=='go_omit',day<10|between(day,13,15),between(time,110,140)) |> 
  mutate(condition = case_when(
    day<5~'cond',
    day>10~'ext',
    .default = group
  )) |> 
  group_by(day,mouse,trial,condition) |> 
  reframe(max_sig = sum(data)) |> 
  group_by(day,condition,mouse) |> 
  reframe(mean_sig = mean(max_sig)) |> 
  group_by(day,condition) |> 
  reframe(mda = mean(mean_sig),semda = sd(mean_sig)/sqrt(n())) |> 
  ggplot() + aes(day,mda,color=condition,ymin = mda-semda,ymax = mda+semda) +
  geom_point() +
  geom_line() +
  geom_errorbar(width=0) + theme_cowplot()

#Fig 3j and k

labelled_photo_data |> 
  filter(trialtype=='c_reward'|trialtype=='c_omit') |> 
  group_by(day,mouse,time,trialtype) |> 
  reframe(mean_sig = mean(data)) |> 
  group_by(day,trialtype,time) |> 
  reframe(da = mean(mean_sig)) |> 
  ggplot()+aes(time,da,color=factor(day))+
  geom_line()+
  facet_grid(.~trialtype) + theme_cowplot()