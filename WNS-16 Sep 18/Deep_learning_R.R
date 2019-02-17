# Load libraries
#install.packages("recipe")
#install.packages("caret")
#install.packages("keras")
#install.packages("lime")
#install.packages("tidyquant")
#install.packages("rsample")
#install.packages("yardstick")
#install.packages("corrr")

library(caret)
library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

#install_keras()

#Select the file with imputed values
churn_data_raw  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)

# Retrieve train and test sets
train_tbl <- churn_data_raw[which(churn_data_raw$dataset=='train'), ]
train_tbl <- train_tbl %>%
  select(-dataset) %>%
  select(-employee_id) %>%
  drop_na() %>%
  select(is_promoted, everything())

test_tbl  <- churn_data_raw[which(churn_data_raw$dataset=='test'), ]
test_tbl <- test_tbl %>%
  select(-dataset) %>%
  select(-employee_id) %>%
  #drop_na() %>%
  select(is_promoted, everything())

# Create recipe
rec_obj <- recipe(is_promoted ~ ., data = train_tbl) %>%
  step_discretize(length_of_service, options = list(cuts = 5)) %>%
  #step_log(avg_training_score) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

# Print the recipe object
rec_obj

# Predictors
x_train_tbl <- bake(rec_obj, newdata = train_tbl) %>% select(-is_promoted)
x_test_tbl  <- bake(rec_obj, newdata = test_tbl) %>% select(-is_promoted)

y_train_vec <- ifelse(pull(train_tbl, is_promoted) == 1, 1, 0)

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 10, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 10, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

keras_model

# Fit the keras model to the training data
history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 20,
  validation_split = 0.30
)

# Print a summary of the training history
print(history)

# Plot the training/validation history of our Keras model
plot(history)

# Predicted Class
yhat_keras_class_vec_prediction <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec_probability  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  estimate   = as.factor(yhat_keras_class_vec_prediction) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec_probability
)

write.csv(estimates_keras_tbl, "estimates_keras_tbl.csv")
