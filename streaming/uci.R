# UCI benchmark datasets

# Longley's Economic Regression Data
data(longley) # regression
# longley %>% write_csv("longley.csv")
# longley %>% select(-Employed) %>% write_csv("longley_train.csv")
# longley %>% rename(target=Employed) %>% select(target) %>% write_csv("longley_train.labels")

library(mlbench)

# data("BostonHousing") # regression
# BostonHousing %>% write_csv("BostonHousing.csv")
# BostonHousing %>% select(-medv) %>% write_csv("BostonHousing_train.csv")
# BostonHousing %>% rename(target=medv) %>% select(target) %>% write_csv("BostonHousing_train.labels")

data("BreastCancer")
BreastCancer %>% write_csv("BreastCancer.csv")
BreastCancer %>% select(-Class, -Id) %>% write_csv("BreastCancer_train.csv")
BreastCancer %>% mutate(target=as.numeric(Class)) %>% select(target) %>% write_csv("BreastCancer_train.labels")

data("Ionosphere")
Ionosphere %>% write_csv("Ionosphere.csv")
Ionosphere %>% select(-Class) %>% write_csv("Ionosphere_train.csv")
Ionosphere %>% mutate(target=as.numeric(Class)) %>% select(target) %>% write_csv("Ionosphere_train.labels")

data("PimaIndiansDiabetes")
PimaIndiansDiabetes %>% write_csv("PimaIndiansDiabetes.csv")
PimaIndiansDiabetes %>% select(-diabetes) %>% write_csv("PimaIndiansDiabetes_train.csv")
PimaIndiansDiabetes %>% mutate(target=as.numeric(diabetes)) %>% select(target) %>% write_csv("PimaIndiansDiabetes_train.labels")

# library(AppliedPredictiveModeling)
# data("abalone")
# abalone %>% write_csv("abalone.csv") # regression
# abalone %>% select(-Rings) %>% write_csv("abalone_train.csv")
# abalone %>% rename(target=Rings) %>% select(target) %>% write_csv("abalone_train.labels")
