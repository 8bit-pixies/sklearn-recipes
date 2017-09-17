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
prostate %>% write_csv("prostate.csv")
leukemia %>% write_csv("leukemia.csv")
lung_cancer %>% write_csv("lung_cancer.csv")

