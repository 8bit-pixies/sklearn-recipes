library(tidyverse)

# open data...
load("data/microarray/alon.RData")
load("data/microarray/singh.RData")
load("data/microarray/golub.RData")
load("data/microarray/gordon.RData")

colon <- alon$x %>% as.data.frame %>% mutate(y=alon$y)
prostate <- singh$x %>% as.data.frame %>% mutate(y=singh$y)
leukemia <- golub$x %>% as.data.frame %>% mutate(y=golub$y)
lung_cancer <- gordon$x %>% as.data.frame %>% mutate(y=gordon$y)

colon %>% write_csv("microarray/colon.csv")
colon %>% select(-y) %>% write_csv("microarray/colon_train.csv")
colon %>% mutate(target=as.numeric(as.factor(y))) %>% select(target) %>% write_csv("microarray/colon_train.labels")

prostate %>% write_csv("microarray/prostate.csv")
prostate %>% select(-y) %>% write_csv("microarray/prostate_train.csv")
prostate %>% mutate(target=as.numeric(as.factor(y))) %>% select(target) %>% write_csv("microarray/prostate_train.labels")

leukemia %>% write_csv("microarray/leukemia.csv")
leukemia %>% select(-y) %>% write_csv("microarray/leukemia_train.csv")
leukemia %>% mutate(target=as.numeric(as.factor(y))) %>% select(target) %>% write_csv("microarray/leukemia_train.labels")

lung_cancer %>% write_csv("microarray/lung_cancer.csv")
lung_cancer %>% select(-y) %>% write_csv("microarray/lung_cancer_train.csv")
lung_cancer %>% mutate(target=as.numeric(as.factor(y))) %>% select(target) %>% write_csv("microarray/lung_cancer_train.labels")
