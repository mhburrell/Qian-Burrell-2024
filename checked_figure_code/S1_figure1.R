# Figure S1

# Fig S1a

lick_data |>
  filter(day %in% c(4, 9), trialtype %in% c("go", "go_omit")) |>
  left_join(mouse_info) |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  group_by(group, day, mouse, trialnumber) |>
  reframe(anticipatory_licks = 1 / 3.5 * sum(between(lick_offset, 0, 3.5))) -> anticipatory_licks

anticipatory_licks |>
  group_by(group, day, mouse) |>
  summarize(mean_anticipatory_licks = mean(anticipatory_licks)) -> s1a_points

s1a_points |>
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_anticipatory_licks), pop_sem = sd(mean_anticipatory_licks) / sqrt(n())) -> s1a_means

s1a_means |> ggplot() +
  aes(day + 1, pop_mean) +
  geom_col() +
  facet_grid(~group) +
  geom_errorbar(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), width = 1) +
  geom_point(data = s1a_points, aes(day + 1, mean_anticipatory_licks), shape = 1) +
  geom_line(data = s1a_points, aes(day + 1, mean_anticipatory_licks, group = mouse), alpha = .5) +
  theme_cowplot(font_size=8) +
  scale_x_continuous(breaks = c(5, 10)) +
  scale_y_continuous(breaks = c(0, 5)) +
  xlab("") +
  ylab("Anticipatory licks (/s)") -> f1a

# stats
anticipatory_licks |> mutate(stat_group = paste(group, day, sep = "_")) -> s1a_stats
lmer(anticipatory_licks ~ stat_group + (1 | mouse / day), data = s1a_stats) -> s1a_model
emmeans(s1a_model, list(pairwise ~ stat_group), adjust = "tukey")

# Fig S1b

lick_data |>
  filter(day %in% c(4, 9), !trialtype %in% c("background","unpred_rew")) |>
  left_join(mouse_info) |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  group_by(group, day, mouse, trialnumber) |>
  reframe(background_licks = sum(between(lick_offset, -1, 0))) -> iti_licks

iti_licks |>
  group_by(group, day, mouse) |>
  summarize(mean_iti_licks = mean(background_licks)) -> s1b_points

s1b_points |>
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_iti_licks), pop_sem = sd(mean_iti_licks) / sqrt(n())) -> s1b_means

s1b_means |> ggplot() +
  aes(day + 1, pop_mean) +
  geom_col() +
  facet_grid(~group) +
  geom_errorbar(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), width = 1) +
  geom_point(data = s1b_points, aes(day + 1, mean_iti_licks), shape = 1) +
  geom_line(data = s1b_points, aes(day + 1, mean_iti_licks, group = mouse), alpha = .5) +
  theme_cowplot(font_size=8) +
  scale_x_continuous(breaks = c(5, 10)) +
  scale_y_continuous(breaks = c(0, 1, 2, 3), limits = c(0, 3)) +
  xlab("") +
  ylab("Background licks (/s)") -> f1b


# stats
iti_licks |> mutate(stat_group = paste(group, day, sep = "_")) -> s1b_stats
lmer(background_licks ~ stat_group + (1 | mouse / day), data = s1b_stats) -> s1b_model
emmeans(s1b_model, list(pairwise ~ stat_group), adjust = "tukey")

# Fig S1c

lick_data |>
  filter(day %in% c(4, 9), trialtype %in% c("go_omit")) |>
  left_join(mouse_info) |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  group_by(group, day, mouse, trialnumber) |>
  reframe(lick_latency = min(lick_offset[lick_offset > 0], 100)) |>
  filter(lick_latency < 100) -> lick_latency

lick_latency |>
  group_by(group, day, mouse) |>
  summarize(mean_lick_latency = mean(lick_latency)) -> s1c_points

s1c_points |>
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_lick_latency), pop_sem = sd(mean_lick_latency) / sqrt(n())) -> s1c_means

s1c_means |> ggplot() +
  aes(day + 1, pop_mean) +
  geom_col() +
  facet_grid(~group) +
  geom_errorbar(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), width = 1) +
  geom_point(data = s1c_points, aes(day + 1, mean_lick_latency), shape = 1) +
  geom_line(data = s1c_points, aes(day + 1, mean_lick_latency, group = mouse), alpha = .5) +
  theme_cowplot(font_size=8) +
  scale_x_continuous(breaks = c(5, 10)) +
  scale_y_continuous(breaks = c(0, 1, 2, 3)) +
  xlab("") +
  ylab("Latency (s)") -> f1c

# stats
lick_latency |> mutate(stat_group = paste(group, day, sep = "_")) -> s1c_stats

lmer(lick_latency ~ stat_group + (1 | mouse / day), data = s1c_stats) -> s1c_model
emmeans(s1c_model, list(pairwise ~ stat_group), adjust = "tukey")

# Fig S1d
lick_summary |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  filter(trialtype %in% c("go", "go_omit"), day %in% c(4, 9)) |>
  group_by(group, day, mouse) |>
  reframe(fraction_correct = mean(anticipatory_licks > 0)) -> s1d_points

s1d_points |>
  group_by(group, day) |>
  reframe(pop_mean = mean(fraction_correct), pop_sem = sd(fraction_correct) / sqrt(n())) -> s1d_means

s1d_means |> ggplot() +
  aes(day + 1, pop_mean) +
  geom_col() +
  facet_grid(~group) +
  geom_errorbar(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), width = 1) +
  geom_point(data = s1d_points, aes(day + 1, fraction_correct), shape = 1) +
  geom_line(data = s1d_points, aes(day + 1, fraction_correct, group = mouse), alpha = .5) +
  theme_cowplot(font_size=8) +
  scale_x_continuous(breaks = c(5, 10)) +
  scale_y_continuous(breaks = c(0, 1)) +
  xlab("") +
  ylab("Fraction correct") -> f1d

# stats

lick_summary |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  filter(trialtype %in% c("go", "go_omit"), day %in% c(4, 9)) |>
  group_by(group, day, mouse, trialnumber) |>
  mutate(correct = anticipatory_licks > 0) |>
  mutate(stat_group = paste(group, day, sep = "_")) -> s1d_stats

glm(correct ~ factor(day) + mouse, data = filter(s1d_stats, group == "degradation"), family = binomial) |> tidy()
glm(correct ~ factor(day) + mouse, data = filter(s1d_stats, group == "conditioning"), family = binomial) |> tidy()
glm(correct ~ factor(day) + mouse, data = filter(s1d_stats, group == "cuedrew"), family = binomial) |> tidy()

# Fig S1e

lick_summary |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  filter(trialtype %in% c("go", "go_omit"), day < 10) |>
  group_by(group, day, mouse) |>
  mutate(block = cut_number(trialnumber, 3, labels = F)) |>
  group_by(group, day, mouse, block) |>
  reframe(anticipatory_licks = mean(anticipatory_licks)*1/3.5) |>
  group_by(group, day, block) |>
  reframe(pop_mean = mean(anticipatory_licks), pop_sem = sd(anticipatory_licks) / sqrt(n())) |> 
  mutate(day_block = day+1 + 0.25*block) -> s1e_points

s1e_points |> ggplot() +
  aes(day_block, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot(font_size=8) +
  facet_grid(~day,scales='free_x') +
  theme(legend.position = "none",strip.background = element_blank(),strip.text.x = element_blank()) +
  xlab("") +
  ylab("Anticipatory licks (/s)") +
  scale_x_continuous(breaks=seq(1.5,10.5),labels = seq(1,10)) +
  scale_y_continuous(limits=c(0,6),breaks=seq(0,6)) -> f1e

# Fig S1f


lick_data |>
  filter(day<10, trialtype != "background") |>
  left_join(mouse_info) |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  group_by(group, day, mouse, trialnumber) |>
  reframe(background_licks = sum(between(lick_offset, -1, 0))) |>
  group_by(group, day, mouse) |>
  reframe(mean_iti_licks = mean(background_licks)) |> 
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_iti_licks), pop_sem = sd(mean_iti_licks) / sqrt(n())) |> 
  mutate(phase = cut(day+1,c(0,5.5,10.5),labels=FALSE))-> s1f_points

s1f_points |> ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot(font_size=8) +
  facet_grid(.~phase,scales='free_x') +
  scale_x_continuous(breaks = c(1,5, 10)) +
  scale_y_continuous(breaks = c(0, 1, 2), limits = c(-0.2, 2)) +
  xlab("Session") +
  ylab("Background licks (/s)") +
  theme(legend.position = "none") -> f1f

# Fig S1g

lick_data |>
  filter(day<10, trialtype %in% c("go_omit")) |>
  left_join(mouse_info) |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  group_by(group, day, mouse, trialnumber) |>
  reframe(lick_latency = min(lick_offset[lick_offset > 0], 100)) |>
  filter(lick_latency < 100) |> 
  group_by(group, day, mouse) |>
  reframe(mean_lick_latency = mean(lick_latency)) |> 
  group_by(group, day) |>
  reframe(pop_mean = mean(mean_lick_latency), pop_sem = sd(mean_lick_latency) / sqrt(n())) |> 
  mutate(phase = cut(day+1,c(0,5.5,10.5),labels=FALSE))-> s1g_points

s1g_points |> ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot(font_size=8) +
  facet_grid(.~phase,scales='free_x') +
  scale_x_continuous(breaks = c(1,5, 10)) +
  scale_y_continuous(limits=c(0,6),breaks=seq(0,6,by=2)) +
  xlab("Session") +
  ylab("Latency (s)") +
  theme(legend.position = "none") -> f1g

# Fig S1h

lick_summary |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  filter(trialtype %in% c("go", "go_omit"), day < 10) |>
  mutate(correct = anticipatory_licks > 0) |>
  group_by(group, day, mouse) |>
  reframe(fraction_correct = mean(correct)) |>
  group_by(group, day) |>
  reframe(pop_mean = mean(fraction_correct), pop_sem = sd(fraction_correct) / sqrt(n())) |> 
  mutate(phase = cut(day+1,c(0,5.5,10.5),labels=FALSE))-> s1h_points

s1h_points |> ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot(font_size=8) +
  facet_grid(.~phase,scales='free_x') +
  scale_x_continuous(breaks = c(1,5, 10)) +
  scale_y_continuous(limits=c(0,1),breaks=seq(0,1,by=0.2)) +
  xlab("Session") +
  ylab("Fraction correct") +
  theme(legend.position = "none") -> f1h

# Fig S1i

lick_summary |>
  mutate(group = fct(group, levels = c("degradation", "conditioning", "cuedrew"))) |>
  filter(trialtype %in% c("go", "go_omit","no_go")) |> 
  mutate(trialtype = if_else(trialtype == "no_go", "no_go", "go")) |> 
  group_by(group, day, trialtype,mouse) |> 
  reframe(mean_ant_licks = mean(anticipatory_licks)*1/3.5) |>
  group_by(group, day, trialtype) |>
  reframe(pop_mean = mean(mean_ant_licks), pop_sem = sd(mean_ant_licks) / sqrt(n())) |> 
  mutate(group_trial = paste(group, trialtype, sep = "_")) |> 
  mutate(phase = cut(day+1,breaks=c(0,5.5,10.5,13.5,16.5,20),labels = FALSE)) -> s1i_points

s1i_points |> ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = group_trial) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  facet_grid(~phase,scales='free_x',space='free_x') +
  theme_cowplot(font_size=8) +
  scale_x_continuous(breaks=seq(1,18))+
  scale_y_continuous(breaks=c(0,5))+
  xlab("Session") +
  ylab("Anticipatory licks (/s)") +
  theme(legend.position = "none") -> f1i

# Fig S1j

lick_summary |> 
  filter(group=='cuedrew',trialtype %in% c("go", "go_omit","c_reward","c_omit")) |>
  mutate(trialtype = if_else(str_detect(trialtype,"c_"),'C','A')) |>
  group_by(day,trialtype,mouse) |>
  reframe(mean_ant_licks = mean(anticipatory_licks)*1/3.5) |>
  group_by(day,trialtype) |>
  reframe(pop_mean = mean(mean_ant_licks), pop_sem = sd(mean_ant_licks) / sqrt(n())) |> 
  mutate(phase=cut(day+1,breaks=c(0,5.5,10.5,13.5,16.5,20),labels = FALSE))-> s1j_points

s1j_points |> ggplot() +
  aes(day + 1, pop_mean, ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem, color = trialtype) +
  geom_point() +
  geom_line() +
  geom_errorbar(width = 0) +
  theme_cowplot(font_size=8) +
  facet_grid(~phase,scales='free_x',space='free_x') +
  scale_x_continuous(breaks=seq(1,10))+
  scale_y_continuous(breaks=seq(0,5),limits = c(0,5))+
  xlab("Session") +
  ylab("Anticipatory licks (/s)") +
  theme(legend.position = "none") -> f1j

# Fig S1k

lick_data |>
  left_join(mouse_info) |>
  filter(day %in% c(4, 9), trialtype %in% c("go", "no_go","unpred_water"),group=='degradation') |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  mutate(lick_bin = cut(lick_offset, breaks = seq(-2, 7, by = 0.2), labels = seq(-1.9, 6.9, 0.2))) |>
  filter(!is.na(lick_bin)) |>
  group_by(mouse, day, trialtype, lick_bin, group) |>
  reframe(lick_rate = 5 * n() / mean(n_trials)) |>
  group_by(group, trialtype, lick_bin, day) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  mutate(plot_group = paste(trialtype,day,sep = "_")) |>
  ggplot() +
  aes(as.numeric(as.character(lick_bin)), mean_lick_rate, color = plot_group, ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA, aes(group = plot_group)) +
  scale_x_continuous(breaks = seq(-2, 6, by = 1)) +
  scale_y_continuous(breaks = c(0, 5,10,15),limits=c(0,15)) +
  theme_cowplot(font_size=8) +
  theme(legend.position = "top") +
  xlab("Time (s)") +
  ylab("Lick rate (/s)") -> f1k

# Fig S1l

lick_data |>
  left_join(mouse_info) |>
  filter(day %in% c(4, 9), trialtype %in% c("go", "no_go","c_reward"),group=='cuedrew') |>
  group_by(mouse, day, trialtype) |>
  mutate(n_trials = n_distinct(trialnumber)) |>
  mutate(lick_bin = cut(lick_offset, breaks = seq(-2, 7, by = 0.2), labels = seq(-1.9, 6.9, 0.2))) |>
  filter(!is.na(lick_bin)) |>
  group_by(mouse, day, trialtype, lick_bin, group) |>
  reframe(lick_rate = 5 * n() / mean(n_trials)) |>
  group_by(group, trialtype, lick_bin, day) |>
  reframe(mean_lick_rate = mean(lick_rate), sem_lick_rate = sd(lick_rate) / sqrt(n())) |>
  mutate(plot_group = paste(trialtype,day,sep = "_")) |>
  ggplot() +
  aes(as.numeric(as.character(lick_bin)), mean_lick_rate, color = plot_group, ymin = mean_lick_rate - sem_lick_rate, ymax = mean_lick_rate + sem_lick_rate) +
  geom_line() +
  geom_ribbon(alpha = 0.1, color = NA, aes(group = plot_group)) +
  scale_x_continuous(breaks = seq(-2, 6, by = 1)) +
  scale_y_continuous(breaks = c(0, 5,10,15),limits=c(0,15)) +
  theme_cowplot(font_size=8) +
  theme(legend.position = "top") +
  xlab("Time (s)") +
  ylab("Lick rate (/s)") -> f1l

#figure composite

plot_grid(f1a,f1b,f1c,f1d,f1e,f1f,f1g,f1h,nrow = 2,align = 'hv',axis='lb') -> fa_h
plot_grid(f1i,f1j,nrow=1,rel_widths = c(2.8,1.2),align='hv',axis='lb') -> fi_j
plot_grid(f1k,f1l,nrow=1,align='hv',axis='lb') -> fk_l

plot_grid(fa_h,fi_j,fk_l,nrow=3,align='hv',axis='lb',rel_heights = c(2,1,1)) -> fig_s1

#save to pdf, portrait letter
ggsave("figS1.pdf",fig_s1,width=8.5,height=11,units='in')