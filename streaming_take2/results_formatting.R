#pandoc tables...
library(pander)
library(tidyverse)
library(reshape2)
options(stringAsFactor=F)

results <- read_csv("results_20170927.csv")
results_melt <- results %>%
  tibble::rownames_to_column() %>%
  mutate(Datasets = stringr::str_c(stringr::str_pad(rowname, 2, pad="0"), "_", Datasets)) %>%
  select(-rowname) %>%
  melt %>%
  mutate(variable = as.character(variable)) 
  

results_dim <- results_melt[endsWith(results_melt$variable, "dim"),] %>%
  mutate(variable = stringr::str_replace(variable, "_dim", ""), 
         metric="#dim")
results_acc <- results_melt[endsWith(results_melt$variable, "acc"),] %>%
  mutate(variable = stringr::str_replace(variable, "_acc", ""),
         metric="acc")

result_melted <- rbind(results_dim, results_acc)
result_melted %>% write_csv("results_20170927_m.csv")

result_melted %>%
  dcast(Datasets+metric~variable) %>%
  arrange(metric, Datasets) %>%
  mutate(Datasets = stringr::str_replace(Datasets, "[0-9]+_", "") ) %>%
  pandoc.table

