#run on cluster

require(tidyverse)
require(arrow)
require(duckdb)
duck_con <- dbConnect(duckdb::duckdb(dbdir = "/scratch/duck1.db"))
dbExecute(duck_con, "PRAGMA memory_limit='58GB'")

dbExecute(duck_con, "CREATE VIEW results2 AS SELECT * FROM read_parquet('/n/netscratch/uchida_users/Lab/mburrell/final_qian/*.parquet')")
results <- tbl(duck_con, "results2")

dbExecute(duck_con, "CREATE VIEW experiment AS SELECT * FROM read_parquet('/n/netscratch/uchida_users/Lab/mburrell/mt_dist2/simulated_trials_anccr.parquet')")
experiment <- tbl(duck_con, "experiment")

experiment |> filter(events==2) |> group_by(ephase,rep) |> 
  slice_max(n=200,order_by = times) -> experiment_extract

experiment_extract |> left_join(results) |> na.omit() |> 
  group_by(ephase,rep,p) |> 
  summarise(mda = mean(DA),sdda=sd(DA)) |> collect() -> anccr_results

anccr_results |> write_parquet('anccr_final_results.parquet')

#run local 

anccr_results <- read_parquet('anccr_final_results.parquet')
param_table <- read_parquet('anccr_param_table.parquet') |> rename(p = p_id)

anccr_results |> 
  mutate(bad_combo = if_else(abs(mda)>5|sdda>2,1,0)) |>
  group_by(p) |> 
  reframe(bad_combo = sum(bad_combo)) |> 
  filter(bad_combo > 3) |> select(p) |> distinct() |> pull(p) -> bad_p

anccr_results |> select(-sdda) |> mutate(rep_n = case_when(
  rep<26~rep,
  between(rep,25,50)~rep-25,
  between(rep,51,75)~rep-50,
), testgroup = case_when(
  rep<26~1,
  between(rep,26,50)~2,
  between(rep,51,75)~3,
)) -> anccr_results

anccr_results |> ungroup() |> select(-rep) |> pivot_wider(names_from = c(testgroup,ephase),values_from = mda,names_glue = "mda_{testgroup}_{ephase}") -> anccr_results

anccr_results |> na.omit() |> mutate(deg = mda_2_2/mda_2_1,cuedrew = mda_3_2/mda_3_1) |> select(p,deg,cuedrew,rep_n) |> 
  group_by(p) |> 
  reframe(mdeg = mean(deg),mcuedrew = mean(cuedrew),sddeg = sd(deg),sdcuedrew = sd(cuedrew)) |>
  filter(sddeg<0.75 & sdcuedrew<0.75) |>
  filter(!(p %in% bad_p)) |> left_join(param_table) |>
  ggplot()+aes(mdeg,mcuedrew,color=factor(w))+geom_point()+xlim(-1,2)+ylim(-1,2)