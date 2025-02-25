require(tidyverse)
require(cowplot)
# Figure 1d

lick_data |>
  left_join(mouse_info) |>
  filter(day %in% c(4, 9), trialtype %in% c("go_omit", "no_go")) |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  mutate(lick_bin = cut(lick_offset, breaks = seq(-2, 7, by = 0.2), labels = seq(-1.9, 6.9, 0.2))) |>
  filter(!is.na(lick_bin)) |>
  group_by(mouse, day, trialtype, lick_bin, group) |>
  reframe(lick_rate = 5 * n() / mean(n_trials)) |>
  group_by(group, trialtype, lick_bin, day) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  mutate(plot_group = case_when(
    day == 4 & trialtype == "go_omit" ~ "A",
    day == 4 & trialtype == "no_go" ~ "B",
    day == 9 & trialtype == "go_omit" ~ "C",
    day == 9 & trialtype == "no_go" ~ "D"
  )) |>
  ggplot() +
  aes(as.numeric(as.character(lick_bin)), mean_lick_rate, color = plot_group, ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA, aes(group = plot_group)) +
  scale_x_continuous(breaks = seq(-2, 6, by = 2)) +
  scale_y_continuous(breaks = c(0, 5)) +
  facet_grid(group ~ .) +
  theme_cowplot() +
  theme(legend.position = "top") +
  xlab("Time (s)") +
  ylab("Lick rate (/s)")

# Figure 1e

lick_data |>
  left_join(mouse_info) |>
  filter(trialtype %in% c("go", "go_omit", "no_go"), day < 10) |>
  mutate(trialtype = ifelse(trialtype == "no_go", "no_go", "go")) |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  filter(between(lick_offset, 0, 3.5)) |>
  group_by(mouse, day, trialtype, group) |>
  reframe(lick_rate = (1 / 3.5) * n() / mean(n_trials)) -> mouse_day_stats

mouse_day_stats |>
  group_by(group, trialtype, day) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  mutate(plot_group = paste(group, trialtype, sep = "_")) |>
  ggplot() +
  aes(day + 1, mean_lick_rate, ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate, color = plot_group) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot() +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = c(1, 5, 10)) +
  xlab("Session number") +
  ylab("Anticipatory licks (/s)") +
  scale_y_continuous(breaks = c(0, 5))

# Figure 1f

mouse_day_stats |> mutate(group = fct(group, levels = c("conditioning", "degradation", "cuedrew"))) -> mouse_day_stats
mouse_day_stats |> filter(day == 9, trialtype == "go") -> fig1f_points
mouse_day_stats |>
  filter(day == 9, trialtype == "go") |>
  group_by(group) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) -> fig1f_stats

fig1f_stats |> ggplot() +
  aes(group, mean_lick_rate, fill = group) +
  geom_col() +
  geom_errorbar(width = 0.33, aes(ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate)) +
  geom_point(data = fig1f_points, aes(x = group, y = lick_rate), shape = 1) +
  theme_cowplot() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Anticipatory licks (/s)") +
  scale_y_continuous(breaks = c(0, 5))

# Figure 1f stats
lick_data |>
  left_join(mouse_info) |>
  filter(trialtype %in% c("go", "go_omit"), day == 9) |>
  mutate(anticipatory_licks = ifelse(between(lick_offset, 0, 3.5), 1, 0)) |>
  group_by(mouse, trialtype, group, trialnumber) |>
  reframe(anticipatory_licks = sum(anticipatory_licks)) -> fig1f_anova_data


# parametric mixed model
require(lmerTest)
require(emmeans)

lmer(anticipatory_licks ~ group + (1 | mouse) + 0, data = fig1f_anova_data, REML = F) -> fig1f_model
emmeans(fig1f_model, list(pairwise ~ group), adjust = "tukey")

# alternative nonparametric
kruskal.test(lick_rate ~ group, data = fig1f_points)
pairwise.wilcox.test(fig1f_points$lick_rate, fig1f_points$group, p.adjust.method = "holm")