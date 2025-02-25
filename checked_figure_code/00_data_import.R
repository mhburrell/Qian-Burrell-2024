require(tidyverse)
require(arrow)

# Load the data

read_parquet("./data/lick_data.parquet") |> select(-`__index_level_0__`)  -> lick_data
read_parquet("./data/trial_data.parquet") |> select(-`__index_level_0__`)  -> trial_data
read_csv("./data/mouse_table.csv",show_col_types = FALSE) -> mouse_info


trial_data |>
  rowwise() |> 
  mutate(odor_on = min(go_odor_on, nogo_odor_on, control_odor_on,1000, na.rm = T)) |>
  ungroup() |> 
  mutate(odor_on = if_else(odor_on==1000,NA,odor_on)) |>
  select(mouse, day, trialnumber, trialtype, odor_on, trial_end) |> 
  group_by(mouse,day) |> 
  fill(odor_on,.direction='downup') |> ungroup() -> trial_data

lick_data |>
  select(mouse, day, trialnumber, trialtype, lick) |>
  left_join(trial_data) |>
  mutate(lick_offset = lick - odor_on) -> lick_data

lick_data |>
  group_by(mouse, day, trialnumber, trialtype) |>
  reframe(
    all_licks = n_distinct(lick_offset) - sum(lick_offset < (-90)),
    pre_licks = sum(between(lick_offset, -2, 0)),
    anticipatory_licks = sum(between(lick_offset, 0, 3.5)),
    post_licks = sum(lick_offset > 3.5),
    lick_latency = min(lick_offset[lick_offset > 3.5], 103.5) - 3.5
  ) -> lick_summary

lick_summary |>
  group_by(mouse, day, trialtype) |>
  arrange(trialnumber) |>
  mutate(trial = as.integer(row_number() - 1)) |>
  ungroup() |> left_join(mouse_info) -> lick_summary

read_parquet("./data/photo_data.parquet") |> select(-`__index_level_0__`) -> photo_data

photo_data |>
  filter(site == "LNacS", mouse != "FgDA_C7") |>
  rename(trialtype = trial_type) |> 
  left_join(lick_summary) |> 
  left_join(mouse_info)-> labelled_photo_data

labelled_photo_data |> mutate(
  mouse = fct(mouse),
  trialtype = fct(trialtype),
  group = fct(group),
  sex = fct(sex),
  site = fct(site),
) -> labelled_photo_data

rm(photo_data)
gc()