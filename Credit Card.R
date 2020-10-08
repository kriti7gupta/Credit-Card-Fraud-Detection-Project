

# Credit Card Fraud Detection

# Install all needed libraries if it is not present

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(corrplot)) install.packages("corrplot")

# Loading all needed libraries

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(caret)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(rstudioapi)
library(corrplot)

## Loading the dataset
creditcard <- read.csv("creditcard.csv")

# Exploratory

# Check dimensions

data.frame("Length" = nrow(creditcard), "Columns" = ncol(creditcard)) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

imbalanced <- data.frame(creditcard)

imbalanced$Class = ifelse(creditcard$Class == 0, 'Legal', 'Fraud') %>% as.factor()

# Visualize the proportion between classes

imbalanced %>%
  ggplot(aes(Class)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Proportions between Legal and Frauds Transactions",
       x = "Class",
       y = "Frequency")

# Check for NAs
apply(creditcard, 2, function(x) sum(is.na(x)))

# Set seed for reproducibility

set.seed(1234)

# Remove the "Time" column from the dataset

creditcard$Class <- as.factor(creditcard$Class)
creditcard <- creditcard %>% select(-Time)

# Plotting the Correlation Matrix
creditcard$Class <- as.numeric(creditcard$Class)
corr_plot <- corrplot(cor(creditcard), method = "circle", type = "upper")

# Split the dataset into train, test dataset and cv

creditcard$Class <- as.factor(creditcard$Class)

train_index <- createDataPartition(
  y = creditcard$Class, 
  p = .6, 
  list = FALSE
)

train <- creditcard[train_index,]

test_cv <- creditcard[-train_index,]

test_index <- createDataPartition(
  y = test_cv$Class, 
  p = .5, 
  list = FALSE)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

rm(train_index, test_index, test_cv)

# Analysis and Modeling

# Model 1
# Naive Baseline model that predict always "legal" 

baseline_model <- data.frame(creditcard)
# Set Class al to Legal (0)
baseline_model$Class = factor(0, c(0,1))
# Make predictions
pred <- prediction(
  as.numeric(as.character(baseline_model$Class)),as.numeric(as.character(creditcard$Class))
)
# Compute the AUC and AUCPR
auc_val_baseline <- performance(pred, "auc")
auc_plot_baseline <- performance(pred, 'sens', 'spec')
aucpr_plot_baseline <- performance(pred, "prec", "rec")

# Create a dataframe 'results' that contains all metrics 
# obtained by the trained models
results <- data.frame(
  Model = "Naive Baseline ", 
  AUC = auc_val_baseline@y.values[[1]],
  AUCPR = 0
)
# Show results on a table
results %>% 
  kable() %>%
  kable_styling(
    bootstrap_options = 
      c("striped", "hover", "condensed", "responsive"),
    position = "center",
    font_size = 10,
    full_width = FALSE
  )


# Model 2
# Naive Bayes Model

set.seed(1234)
# Build the model with Class as target and all other variables
# as predictors
naive_model <- naiveBayes(Class ~ ., data = train, laplace=1)
# Predict
predictions <- predict(naive_model, newdata=test)
# Compute the AUC and AUCPR for the Naive Model
pred <- prediction(as.numeric(predictions) , test$Class)
auc_val_naive <- performance(pred, "auc")
auc_plot_naive <- performance(pred, 'sens', 'spec')
aucpr_plot_naive <- performance(pred, "prec", "rec")
aucpr_val_naive <- pr.curve(
  scores.class0 = predictions[test$Class == 1], 
  scores.class1 = predictions[test$Class == 0],
  curve = T,  
  dg.compute = T
)
# Adding the respective metrics to the results dataset
results <- results %>% add_row(
  Model = "Naive Bayes", 
  AUC = auc_val_naive@y.values[[1]],
  AUCPR = aucpr_val_naive$auc.integral
)
# Show results on a table
results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed","responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Model 3

#Logistic Regression

set.seed(1234)
glm_model <- glm(Class ~ ., data = train, family = "binomial")

predictions <- predict(glm_model, newdata=test)

pred <- prediction(as.numeric(predictions) , test$Class)

auc_val_glm <- performance(pred, "auc")

auc_plot_glm  <- performance(pred, 'sens', 'spec')
aucpr_plot_glm  <- performance(pred, "prec", "rec")

aucpr_val_glm  <- pr.curve(
  scores.class0 = predictions[test$Class == 1], 
  scores.class1 = predictions[test$Class == 0],
  curve = T,  
  dg.compute = T
)

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "Logistic Regression", 
  AUC = auc_val_glm@y.values[[1]],
  AUCPR = aucpr_val_glm$auc.integral
)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed","responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 

# Model 4

# KNN
# Set seed 1234 for reproducibility
set.seed(1234)
# Build a KNN Model with Class as Target and all other
# variables as predictors. k is set to 5
knn_model <- knn(train[,-30], test[,-30], train$Class, k=5, prob = TRUE)
# Compute the AUC and AUCPR for the KNN Model
pred <- prediction(
  as.numeric(as.character(knn_model)), as.numeric(as.character(test$Class))
)
auc_val_knn <- performance(pred, "auc")
auc_plot_knn <- performance(pred, 'sens', 'spec')
aucpr_plot_knn <- performance(pred, "prec", "rec")
aucpr_val_knn <- pr.curve(
  scores.class0 = knn_model[test$Class == 1], 
  scores.class1 = knn_model[test$Class == 0],
  curve = T,  
  dg.compute = T
)
# Adding the respective metrics to the results dataset
results <- results %>% add_row(
  Model = "K-Nearest Neighbors ", 
  AUC = auc_val_knn@y.values[[1]],
  AUCPR = aucpr_val_knn$auc.integral
)
# Show results on a table
results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",  "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)
# Model 5

# Random Forest

set.seed(1234)
# Build a Random Forest Model with Class as Target and all other
# variables as predictors. The number of trees is set to 100

rf_model <- randomForest(Class ~ ., data = train, ntree = 100)
# Get the feature importance
feature_imp_rf <- data.frame(importance(rf_model))
# Make predictions based on this model
predictions <- predict(rf_model, newdata=test)
# Compute the AUC and AUPCR
pred <- prediction(
  as.numeric(as.character(predictions)), as.numeric(as.character(test$Class))
)
auc_val_rf <- performance(pred, "auc")
auc_plot_rf <- performance(pred, 'sens', 'spec')
aucpr_plot_rf <- performance(pred, "prec", "rec", curve = T,  dg.compute = T)
aucpr_val_rf <- pr.curve(scores.class0 = predictions[test$Class == 1], scores.class1 = predictions[test$Class == 0],curve = T,  dg.compute = T)


# Adding the respective metrics to the results dataset
results <- results %>% add_row(
  Model = "Random Forest",
  AUC = auc_val_rf@y.values[[1]],
  AUCPR = aucpr_val_rf$auc.integral)
# Show results on a table
results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)
# Show feature importance on a table
feature_imp_rf %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Shows the results
results %>% 
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

