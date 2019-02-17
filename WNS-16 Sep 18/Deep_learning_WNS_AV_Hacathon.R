#https://blogs.rstudio.com/tensorflow/posts/2018-01-11-keras-customer-churn/
#https://blogs.rstudio.com/tensorflow/posts/2018-01-24-keras-fraud-autoencoder/

# Load libraries
install.packages("recipe")
install.packages("caret")
install.packages("keras")
install.packages("lime")
install.packages("tidyquant")
install.packages("rsample")
install.packages("yardstick")
install.packages("corrr")

library(caret)
library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

#install_keras()

churn_data_raw  <- read.csv(file.choose(),header = TRUE , stringsAsFactors = FALSE)

churn_data_raw <- churn_data_raw[which(churn_data_raw$dataset=='train'), ]

glimpse(churn_data_raw)

#churn_data_raw$SeniorCitizen <- as.factor(churn_data_raw$SeniorCitizen)

# Remove unnecessary data
churn_data_tbl <- churn_data_raw %>%
  select(-dataset) %>%
  select(-employee_id) %>%
  drop_na() %>%
  select(is_promoted, everything())

glimpse(churn_data_tbl)

# Split test/training sets
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.60)
train_test_split

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

# Determine if log transformation improves correlation 
# between TotalCharges and Churn
train_tbl %>%
  select(is_promoted, avg_training_score) %>%
  mutate(
    is_promoted = is_promoted %>% as.factor() %>% as.numeric(),
    LogTotalCharges = log(avg_training_score)
  ) %>%
  correlate() %>%
  focus(is_promoted) %>%
  fashion()

# Create recipe
rec_obj <- recipe(is_promoted ~ ., data = train_tbl) %>%
  step_discretize(length_of_service, options = list(cuts = 6)) %>%
  step_log(avg_training_score) %>%
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
y_test_vec  <- ifelse(pull(test_tbl, is_promoted) == 1, 1, 0)

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 20, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 20, 
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
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl

options(yardstick.event_first = FALSE)

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)
# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)
# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)

# Precision
tibble(
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate)
)

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
####################################### That all is needed to solve##########
class(model_keras)

# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  "classification"
}

# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}

model_type = "classification"

# Test our predict_model() function
predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl, 
  model          = model_keras, 
  bin_continuous = FALSE
)

# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[1:10, ], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
  kernel_width = 0.5
)

plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

# Feature correlations to Churn
corrr_analysis <- x_train_tbl %>%
  mutate(is_promoted = y_train_vec) %>%
  correlate() %>%
  focus(is_promoted) %>%
  rename(feature = rowname) %>%
  arrange(abs(is_promoted)) %>%
  mutate(feature = as_factor(feature)) 
corrr_analysis

# Correlation visualization
corrr_analysis %>%
  ggplot(aes(x = is_promoted, y = fct_reorder(feature, desc(is_promoted)))) +
  geom_point() +
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[2]], 
               data = corrr_analysis %>% filter(is_promoted > 0)) +
  geom_point(color = palette_light()[[2]], 
             data = corrr_analysis %>% filter(is_promoted > 0)) +
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[1]], 
               data = corrr_analysis %>% filter(is_promoted < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = corrr_analysis %>% filter(is_promoted < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  theme_tq() +
  labs(title = "is_promoted Correlation Analysis", subtitle = paste("Positive Correlations (contribute to is_promoted),", "Negative Correlations (prevent is_promoted)"),
       y = "Feature Importance")


