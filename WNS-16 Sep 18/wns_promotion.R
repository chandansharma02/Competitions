#######################################
#       Install Packages              #
#######################################
install.packages("readxl")
install.packages("mice")
install.packages("VIM")
install.packages("dtplyr")
install.packages("caret")
install.packages("caret", dependencies = TRUE)
ninstall.packages("recipes")


library(readxl)
library(mice)
library(VIM)
library(dtplyr) 
library(caret)

#######################################

#######################################
#             Read File               #
#######################################
consolidated_data  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)

consolidated_data_imputed1  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)
consolidated_data_imputed2  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)
consolidated_data_imputed3  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)
consolidated_data_imputed4  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)
consolidated_data_imputed5  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)

#######################################
#       Redifine Col Name             #
#######################################
#colnames(train)[11] = "KPIs_met"
#colnames(train)[12] = "Award_won"

#######################################
#         Deal with NA Value          #
#######################################
# identify location of NAs in vector
which(is.na(consolidated_data))

# identify count of NAs in data frame
sum(is.na(consolidated_data))
colSums(is.na(consolidated_data))

mice_plot <- aggr(consolidated_data, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=FALSE,
                  labels=names(consolidated_data), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

original <- consolidated_data

namesss <- data.frame(names(consolidated_data))

consolidated_data <- consolidated_data %>%
  mutate(
    dataset = as.factor(dataset),
    employee_id = as.factor(employee_id),
    department = as.factor(department),
    region = as.factor(region),
    education = as.factor(education),
    gender = as.factor(gender),
    recruitment_channel = as.factor(recruitment_channel),
    previous_year_rating = as.factor(previous_year_rating),
    KPIs_met = as.factor(KPIs_met),
    is_promoted = as.factor(is_promoted)
  )

#because the dataset is very large lets try to reduce some columns
consolidated_data = consolidated_data[c(3:14)] 

init = mice(consolidated_data1, maxit=0) 
meth = init$method
#Use below to see which method will apply to which variable
test <- data.frame(meth)
#See below metric to chech which variable will be used to predict which missing column
predM = init$predictorMatrix

#The code below will remove the variable as a predictor but still will be imputed. 
#predM[, c("dataset")]=0

#If you want to skip a variable from imputation use the code below. 
#meth[c("is_promoted")]=""

#If you want to change the default method for the data set
#meth[c("Cholesterol")]="norm" 
#meth[c("Smoking")]="logreg" 
#meth[c("Education")]="polyreg"

set.seed(103)
imputed = mice(consolidated_data, method=meth, predictorMatrix=predM, m=5)

consolidated_data_imputed1 <- complete(imputed,1)
consolidated_data_imputed2 <- complete(imputed,2)
consolidated_data_imputed3 <- complete(imputed,3)
consolidated_data_imputed4 <- complete(imputed,4)
consolidated_data_imputed5 <- complete(imputed,5)

write.csv(consolidated_data_imputed1,"F:\\All_Prep\\Competition\\WNS\\imputedset\\consolidated_data_imputed1.csv")
write.csv(consolidated_data_imputed2,"F:\\All_Prep\\Competition\\WNS\\imputedset\\consolidated_data_imputed2.csv")
write.csv(consolidated_data_imputed3,"F:\\All_Prep\\Competition\\WNS\\imputedset\\consolidated_data_imputed3.csv")
write.csv(consolidated_data_imputed4,"F:\\All_Prep\\Competition\\WNS\\imputedset\\consolidated_data_imputed4.csv")
write.csv(consolidated_data_imputed5,"F:\\All_Prep\\Competition\\WNS\\imputedset\\consolidated_data_imputed5.csv")

#After this I have mantually combined original and consolidated data file

#Select which data to use for training, validation and testing
selected_set <- consolidated_data_imputed1

selected_set <- selected_set[which(selected_set$dataset=='train'), ]
selected_set <- selected_set[c(3:15)]

# Divide the data into train and test to check the score
## 75% of the sample size
smp_size <- floor(0.75 * nrow(selected_set))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(selected_set)), size = smp_size)

selected_set_train <- selected_set[train_ind, ]
selected_set_test <- selected_set[-train_ind, ]
selected_set_train <- data.frame(selected_set_train)
selected_set_test <- data.frame(selected_set_test)

# Try model 1 to see the accuracy of the model
#######################################
#              Model Run              #
#######################################

#logistic regression model - not successful
logitMod1 <- glm (is_promoted ~ ., data = selected_set_train, family = binomial(link = 'logit'))
summary(logitMod1)

log_predict <- predict(logitMod1,newdata = selected_set_test,type = "response")
log_predict <- ifelse(log_predict > 0.5,1,0)
log_predict <- data.frame(log_predict)

pr <- prediction(log_predict,selected_set_test$is_promoted)
perf <- performance(pr,measure = "tpr",x.measure = "fpr") 
plot(perf) > auc(selected_set_test$is_promoted,log_predict) #0.76343

# Random forest
library(ROCR) 
library(Metrics)
library(randomForest)

logitMod2 <- randomForest(is_promoted ~ ., data = selected_set_train, importance =TRUE )
rf_predicted <- plogis(predict(logitMod2, selected_set_test))  # predicted scores
plot(logitMod2)

rf_predicted_value <- ifelse(rf_predicted> 0.95,"1","0")
predicted_merge <- cbind(selected_set_test$is_promoted,rf_predicted_value)

