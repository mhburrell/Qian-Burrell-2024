#Fig6 Data Prep
input_folder <- "./rnn_data/"
output_folder <- "./labelled_rnn_data/"

new_rnn <- open_dataset(input_folder, unify_schemas = TRUE)
new_rnn |>
  filter(condition == "degradation", trained == TRUE) |>
  select(seed) |>
  distinct() |>
  collect() -> seed_list

for (i in seed_list$seed) {
  new_rnn |>
    naive_vs_pretrained() |>
    filter(seed == i, naive_group == TRUE) |>
    collect() |>
    label_data() |>
    align_time() |>
    do_cca() |>
    mutate(pca_group = "naive") |>
    group_by(seed, pca_group) |>
    write_dataset(output_folder)
  new_rnn |>
    naive_vs_pretrained() |>
    filter(seed == i, pretrained_group == TRUE) |>
    collect() |>
    label_data() |>
    align_time() |>
    do_cca() |>
    mutate(pca_group = "pretrained") |>
    group_by(seed, pca_group) |>
    write_dataset(output_folder)
}
