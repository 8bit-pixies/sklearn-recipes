library(datamicroarray)
library(tidyverse)
describe_data()

# colon
data('alon', package = 'datamicroarray')

# prostate
data('singh', package = 'datamicroarray')

# leukemia
data('golub', package = 'datamicroarray')

# lung cancer
data('gordon', package = 'datamicroarray')

colon <- alon$x %>% as.data.frame %>% mutate(y=alon$y)
prostate <- singh$x %>% as.data.frame %>% mutate(y=singh$y)
leukemia <- golub$x %>% as.data.frame %>% mutate(y=golub$y)
lung_cancer <- gordon$x %>% as.data.frame %>% mutate(y=gordon$y)

colon %>% write_csv("colon.csv")
colon %>% select(-y) %>% write_csv("colon_train.csv")
colon %>% mutate(target=as.factor(y)) %>% select(target) %>% write_csv("colon_train.labels")

prostate %>% write_csv("prostate.csv")
prostate %>% select(-y) %>% write_csv("prostate_train.csv")
prostate %>% mutate(target=as.factor(y)) %>% select(target) %>% write_csv("prostate_train.labels")

leukemia %>% write_csv("leukemia.csv")
leukemia %>% select(-y) %>% write_csv("leukemia_train.csv")
leukemia %>% mutate(target=as.factor(y)) %>% select(target) %>% write_csv("leukemia_train.labels")

lung_cancer %>% write_csv("lung_cancer.csv")
lung_cancer %>% select(-y) %>% write_csv("lung_cancer_train.csv")
lung_cancer %>% mutate(target=as.factor(y)) %>% select(target) %>% write_csv("lung_cancer_train.labels")
