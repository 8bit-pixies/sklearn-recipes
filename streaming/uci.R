# UCI benchmark datasets
library(tidyverse)
# Longley's Economic Regression Data
data(longley) # regression
# longley %>% write_csv("uci/longley.csv")
# longley %>% select(-Employed) %>% write_csv("uci/longley_train.csv")
# longley %>% rename(target=Employed) %>% select(target) %>% write_csv("uci/longley_train.labels")

library(mlbench)
library(nutshell)

# data("BostonHousing") # regression
# BostonHousing %>% write_csv("uci/BostonHousing.csv")
# BostonHousing %>% select(-medv) %>% write_csv("uci/BostonHousing_train.csv")
# BostonHousing %>% rename(target=medv) %>% select(target) %>% write_csv("uci/BostonHousing_train.labels")

#data("BreastCancer")
BreastCancer %>% write_csv("uci/BreastCancer.csv")
BreastCancer %>% select(-Class, -Id) %>% write_csv("uci/BreastCancer_train.csv")
BreastCancer %>% mutate(target=as.numeric(Class)) %>% select(target) %>% write_csv("uci/BreastCancer_train.labels")

data("Ionosphere")
Ionosphere %>% write_csv("uci/Ionosphere.csv")
Ionosphere %>% select(-Class) %>% write_csv("uci/Ionosphere_train.csv")
Ionosphere %>% mutate(target=as.numeric(Class)) %>% select(target) %>% write_csv("uci/Ionosphere_train.labels")

#data("PimaIndiansDiabetes")
#PimaIndiansDiabetes %>% write_csv("uci/PimaIndiansDiabetes.csv")
#PimaIndiansDiabetes %>% select(-diabetes) %>% write_csv("uci/PimaIndiansDiabetes_train.csv")
#PimaIndiansDiabetes %>% mutate(target=as.numeric(diabetes)) %>% select(target) %>% write_csv("uci/PimaIndiansDiabetes_train.labels")

data("spambase")
# spambase$is_spam
spambase %>% write_csv("uci/spambase.csv")
spambase %>% select(-is_spam) %>% write_csv("uci/spambase_train.csv")
spambase %>% mutate(target=as.numeric(is_spam)) %>% select(target) %>% write_csv("uci/spambase_train.labels")

SPECTF <- createdatasets::createSPECTF()
SPECTF %>% write_csv("uci/spectf.csv")
SPECTF %>% select(-Diagnosis) %>% write_csv("uci/spectf_train.csv")
SPECTF %>% mutate(target=as.numeric(Diagnosis)) %>% select(target) %>% write_csv("uci/spectf_train.labels")

wdbc <- createdatasets::createWDBC()
wdbc %>% write_csv("uci/wdbc.csv")
wdbc %>% select(-ID, -Diagnosis) %>% write_csv("uci/wdbc_train.csv")
wdbc %>% mutate(target=as.numeric(Diagnosis)) %>% select(target) %>% write_csv("uci/wdbc_train.labels")


# library(AppliedPredictiveModeling)
# data("abalone")
# abalone %>% write_csv("uci/abalone.csv") # regression
# abalone %>% select(-Rings) %>% write_csv("uci/abalone_train.csv")
# abalone %>% rename(target=Rings) %>% select(target) %>% write_csv("uci/abalone_train.labels")
