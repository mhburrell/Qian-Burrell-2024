# Figure S2

read_parquet("./data/photo_data.parquet") |> select(-`__index_level_0__`) -> photo_data

photo_data |>
  filter(mouse != "FgDA_C7") |>
  rename(trialtype = trial_type) |>
  left_join(lick_summary) |>
  left_join(mouse_info) -> all_photo_data

all_photo_data |> mutate(
  mouse = fct(mouse),
  trialtype = fct(trialtype),
  group = fct(group),
  sex = fct(sex),
  site = fct(site),
) -> all_photo_data

# exclusions
good_regions <- read_parquet('./data/good_regions.parquet') |> rename(mouse = subject)


all_photo_data |>
  filter(site %in% c("AMOT", "PMOT", "ALOT", "PLOT", "MNacS", "LNacS"), day %in% c(4, 9)) |>
  select(trial, time, day, mouse, data, site, trialtype, trialnumber, group, lick_latency) -> figS2_data

figS2_data |>
  filter(trialtype == "go", lick_latency<0.25) |>
  group_by(day, mouse, time, site, group) |>
  reframe(mean_sig = mean(data)) -> mouse_day_mean

mouse_day_mean|>
  left_join(good_regions, by = join_by(mouse == mouse, site == region)) |>
  filter(good==1) |> 
  group_by(day, time, site, group) |>
  reframe(pop_mean = mean(mean_sig), pop_sem = sd(mean_sig) / sqrt(n())) |>
  ggplot() +
  aes(x = time * 1 / 20 - 2, y = pop_mean, group = site, color = factor(group)) +
  geom_line() +
  geom_ribbon(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), alpha = 0.2, color = NA) +
  facet_grid(site ~ group + day) +
  theme_cowplot() +
  theme(legend.position = "none")

calculate_cosine_similarity <- function(df) {
  data_matrix <- as.matrix(df |> select(AMOT, PMOT, ALOT, PLOT, MNacS, LNacS))
  sim_matrix <- lsa::cosine(data_matrix)
  results <- as_tibble(as.matrix(sim_matrix))
  results$site <- c("AMOT", "PMOT", "ALOT", "PLOT", "MNacS", "LNacS")
  return(results)
}

mouse_day_mean |> 
  pivot_wider(names_from = site, values_from = mean_sig) |>
  group_by(mouse,day) |> 
  nest() |>
  mutate(similarity = map(data, calculate_cosine_similarity)) |>
  select(-data) |>
  unnest(similarity) |>
  pivot_longer(cols = -c(mouse, day, site), names_to = "name", values_to = "value") -> mouse_day_sim





mouse_day_sim |>
  left_join(good_regions, by = join_by(mouse == mouse, name == region)) |>
  rename(good1 = good) |>
  left_join(good_regions, by = join_by(mouse == mouse, site == region)) |>
  mutate(good = good & good1) |>
  filter(good == T) |>  mutate(site = fct(site, levels = c("AMOT", "PMOT", "ALOT", "PLOT", "MNacS", "LNacS")), name = fct(name, levels = rev(c("AMOT", "PMOT", "ALOT", "PLOT", "MNacS", "LNacS"))))  -> mouse_day_sim_filt

mouse_day_sim_filt|>
  filter(day == 4) |>
  group_by(site, name) |>
  reframe(value = mean(value)) |>
  ggplot() +
  aes(site, name, fill = value) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), size = 5) +
  scale_fill_viridis_c(limits = c(0, 1)) -> day5_heatmap


mouse_day_sim_filt |> 
  filter(day == 9) |>
  group_by(site, name) |>
  reframe(value = mean(value)) |>
  ggplot() +
  aes(site, name, fill = value) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), size = 5) +
  scale_fill_viridis_c(limits = c(0, 1)) -> day10_heatmap

# fig 2c

all_photo_data |>
  filter(site %in% c("AMOT", "PMOT", "ALOT", "PLOT", "MNacS", "LNacS"), day<10, lick_latency<0.25, trialtype=='go') |>
  select(trial, time, day, mouse, data, site, trialtype, trialnumber, group) |> 
  left_join(good_regions, by = join_by(mouse == mouse, site == region)) |>
  filter(good==1) |>
  group_by(day, mouse, time, site, group) |>
  reframe(mean_sig = mean(data)) |>
  group_by(day, time, site, group) |>
  reframe(pop_mean = mean(mean_sig), pop_sem = sd(mean_sig) / sqrt(n())) |>
  ggplot() +
  aes(x = time * 1 / 20 - 2, y = pop_mean, group = site, color = factor(group)) +
  geom_line() +
  geom_ribbon(aes(ymin = pop_mean - pop_sem, ymax = pop_mean + pop_sem), alpha = 0.2, color = NA) +
  facet_grid(group+site ~ day) +
  theme_cowplot() +
  theme(legend.position = "none")
