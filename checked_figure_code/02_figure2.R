# figure 2
require(cowplot)
require(pals)

# Figure 2c heat maps

# left panel
labelled_photo_data |>
  filter(trialtype == "go", day < 5, mouse == "FgDA_07") |>
  ggplot() +
  aes((time - 40) * 0.05, -trial, fill = data) +
  geom_tile() +
  facet_grid(day ~ .) +
  theme_cowplot() +
  scale_fill_gradient2(high = "red", low = "blue", midpoint = 0, limits = c(-12.5, 12.5), na.value = "red") +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# middle panel
labelled_photo_data |>
  filter(trialtype == "go", between(day, 5, 9), mouse == "FgDA_07") |>
  ggplot() +
  aes((time - 40) * 0.05, -trial, fill = data) +
  geom_tile() +
  facet_grid(day ~ .) +
  theme_cowplot() +
  scale_fill_gradient2(high = "red", low = "blue", midpoint = 0, limits = c(-12.5, 12.5), na.value = "red") +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# right panel 
labelled_photo_data |>
  filter(trialtype == "go", between(day, 5, 9), mouse == "FgDA_C1") |>
  ggplot() +
  aes((time - 40) * 0.05, -trial, fill = data) +
  geom_tile() +
  facet_grid(day ~ .) +
  theme_cowplot() +
  scale_fill_gradient2(high = "red", low = "blue", midpoint = 0, limits = c(-12.5, 12.5), na.value = "red") +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# fig 2d population averages

# left

labelled_photo_data |>
  filter(trialtype == "go", day < 5) |>
  group_by(mouse, day, time) |>
  reframe(data = mean(data)) |>
  group_by(day, time) |>
  reframe(mean_da = mean(data), sem_da = sd(data) / sqrt(n())) |>
  ggplot() +
  aes((time - 40) * 0.05, mean_da, ymin = mean_da - sem_da, ymax = mean_da + sem_da) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA) +
  facet_grid(day ~ .) +
  theme_cowplot() +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# middle

labelled_photo_data |>
  filter(trialtype == "go", between(day, 5, 9), group == "degradation") |>
  group_by(mouse, day, time) |>
  reframe(data = mean(data)) |>
  group_by(day, time) |>
  reframe(mean_da = mean(data), sem_da = sd(data) / sqrt(n())) |>
  ggplot() +
  aes((time - 40) * 0.05, mean_da, ymin = mean_da - sem_da, ymax = mean_da + sem_da) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA) +
  facet_grid(day ~ .) +
  theme_cowplot() +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# right

labelled_photo_data |>
  filter(trialtype == "go", between(day, 5, 9), group == "cuedrew") |>
  group_by(mouse, day, time) |>
  reframe(data = mean(data)) |>
  group_by(day, time) |>
  reframe(mean_da = mean(data), sem_da = sd(data) / sqrt(n())) |>
  ggplot() +
  aes((time - 40) * 0.05, mean_da, ymin = mean_da - sem_da, ymax = mean_da + sem_da) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA) +
  facet_grid(day ~ .) +
  theme_cowplot() +
  xlab("Time to odor (s)") +
  scale_x_continuous(breaks = c(0, 2, 4, 6), limits = c(-1, 7))

# fig 2e

labelled_photo_data |>
  filter(trialtype %in% c("go", "go_omit"), day < 10, all_licks > 0, between(time, 40, 60),lick_latency<0.25) |>
  group_by(group, mouse, day, trialnumber) |>
  reframe(max_sig = max(data)) -> max_sig_data

max_sig_data |>
  group_by(group, mouse, day) |>
  reframe(mean_max = mean(max_sig)) |>
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_max), pop_sem = sd(mean_max) / sqrt(n())) |>
  ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group) +
  geom_point() +
  geom_line() + theme_cowplot() + 
  geom_errorbar(width = 0) +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = seq(1,10)) +
  xlab("Session number") +
  ylab("Axonal Calcium (z-score)") +
  scale_y_continuous(breaks = c(0, 5,10),limits=c(-0.5,14))

#stats

require(lmerTest)
require(emmeans)

lmer(max_sig~factor(day)*group+(1|mouse)+0,data=max_sig_data,REML = F)  -> max_sig_model
lmer(max_sig~(1|mouse)+0,data=max_sig_data,REML = F)  -> max_sig_null_model
anova(max_sig_null_model,max_sig)
emm_options(pbkrtest.limit = 10000)
pairs(emmeans(max_sig_model,~group|factor(day)))

# fig 2f

max_sig_data |> filter(day%in%c(4,9)) |> 
  mutate(condition = case_when(
    day < 5 ~ "conditioning",
    day > 5 & group == "cuedrew" ~ "cued reward",
    day > 5 & group == "degradation" ~ "degradation"
  )) -> fig2f_stats_data

fig2f_stats_data |>
  group_by(mouse,condition) |> 
  reframe(mean_max = mean(max_sig)) -> fig2f_points

fig2f_points |>
  group_by(condition) |>
  reframe(pop_mean = mean(mean_max), pop_sem = sd(mean_max) / sqrt(n())) |> 
  mutate(condition = fct(condition,levels=c('conditioning','degradation','cued reward')))-> fig2f_stats

fig2f_stats |> ggplot() +
  aes(condition, pop_mean, fill = condition) +
  geom_col() +
  geom_errorbar(width = 0.33, aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem)) +
  geom_point(data = fig2f_points, aes(x = condition, y = mean_max), shape = 1) + 
  geom_line(data = fig2f_points, aes(x = condition, y = mean_max,group=mouse)) +
  theme_cowplot() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Axonal Calcium (z-score)") +
  scale_y_continuous(breaks = c(0, 15),limits=c(-0.5,16))

#stats

lmer(max_sig~condition+(1|mouse)+0,data=fig2f_stats_data,REML = F)  -> fig2f_model
pairs(emmeans(fig2f_model,~condition))