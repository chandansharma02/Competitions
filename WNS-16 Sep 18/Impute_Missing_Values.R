library(mice)
# See below code to learn about MICE fully
# https://www.rdocumentation.org/packages/mice/versions/3.3.0/topics/mice

dat <- read.csv(url("https://goo.gl/4DYzru"), header=TRUE, sep=",")

original <- dat

set.seed(10)
dat[sample(1:nrow(dat), 20), "Cholesterol"] <- NA
dat[sample(1:nrow(dat), 20), "Smoking"] <- NA
dat[sample(1:nrow(dat), 20), "Education"] <- NA
dat[sample(1:nrow(dat), 5), "Age"] <- NA
dat[sample(1:nrow(dat), 5), "BMI"] <- NA

sapply(dat, function(x) sum(is.na(x)))
#install.packages("dtplyr")

library(dtplyr) 
dat <- dat %>%
  mutate(
    Smoking = as.factor(Smoking),
    Education = as.factor(Education),
    Cholesterol = as.numeric(Cholesterol)
  )

init = mice(dat, maxit=0) 
meth = init$method
#Use below to see which method will apply to which variable
test <- data.frame(meth)
#See below metric to chech which variable will be used to predict which missing column
predM = init$predictorMatrix

#The code below will remove the variable as a predictor but still will be imputed. 
predM[, c("BMI")]=0

#If you want to skip a variable from imputation use the code below. 
meth[c("Age")]=""

meth[c("Cholesterol")]="norm" 
meth[c("Smoking")]="logreg" 
meth[c("Education")]="polyreg"

set.seed(103)
imputed = mice(dat, method=meth, predictorMatrix=predM, m=5)

imputed <- complete(imputed)

# Cholesterol
actual <- original$Cholesterol[is.na(dat$Cholesterol)]
predicted <- imputed$Cholesterol[is.na(dat$Cholesterol)]
# Smoking
actual <- original$Smoking[is.na(dat$Smoking)] 
predicted <- imputed$Smoking[is.na(dat$Smoking)] 
table(actual)
table(predicted)
mean(actual) 
mean(predicted)
