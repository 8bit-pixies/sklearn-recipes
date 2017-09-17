# UCI benchmark datasets

# Longley's Economic Regression Data
data(longley) # regression
longley %>% write_csv("longley.csv")

library(mlbench)

data("BostonHousing") # regression
BostonHousing %>% write_csv("BostonHousing.csv")
data("BreastCancer")
BreastCancer %>% write_csv("BreastCancer.csv")
data("Ionosphere")
Ionosphere %>% write_csv("Ionosphere.csv")
data("PimaIndiansDiabetes")
PimaIndiansDiabetes %>% write_csv("PimaIndiansDiabetes.csv")

library(AppliedPredictiveModeling)
data("abalone")
abalone %>% write_csv("abalone") # regression
