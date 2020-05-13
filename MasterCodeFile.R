##### Loading Libraries #####

library(resample)
library(recipes)
library(h2o)
library(tidymodels)
library(dplyr)
library(ggplot2)
library(pdp)
library(vip)
library(iml)
library(DALEX)
library(lime)

h2o.init()

##### Loading and Preprocessing Data #######

# Load the Direct Marketing Data

split <- initial_split(dm,strata = "AmountSpent")
train <- training(split)
test <- testing(split)

#Creating a recipe and preprocessing the data

rec <- recipe(AmountSpent ~ ., data = dm)
rec <- rec %>% step_knnimpute(all_predictors(), neighbors = 5)
rec <- rec %>% step_dummy(all_predictors(), -all_numeric())
rec <- rec %>% step_center(all_predictors()) %>% step_scale(all_predictors())

train_h20 <- prep(rec, training = train) %>% juice() %>% as.h2o()
test_h20 <- prep(rec, training = train) %>% bake(new_data = test) %>% as.h2o()

summary(train_h20)

#getting the names of Xs and Y

Y <- "AmountSpent"

##### Setting up Ensembling #######

# Train & cross-validate a GLM model

#regularized regression
glm <- h2o.glm(y = Y, training_frame = train_h20, alpha = 0.5, remove_collinear_columns = TRUE, nfolds = 10, 
               keep_cross_validation_predictions = TRUE, seed = 123, fold_assignment = "Modulo")

summary(glm)
h2o.performance(glm, newdata = test_h20)

#Random forest
rf <- h2o.randomForest(y = Y, training_frame = train_h20, ntrees = 100, max_depth = 200,
                       nfolds = 10, keep_cross_validation_predictions = TRUE,
                       stopping_rounds = 50, stopping_tolerance = 0, stopping_metric = "RMSE", seed = 123, fold_assignment = "Modulo")

h2o.performance(rf, newdata = test_h20)

#Gradient Boosting

gbm <- h2o.gbm(y = Y, training_frame = train_h20, ntrees = 100, nfolds = 10,
               keep_cross_validation_predictions = TRUE, 
               stopping_rounds = 50, stopping_tolerance = 0, stopping_metric = "RMSE", seed = 123, fold_assignment = "Modulo")

h2o.performance(gbm, newdata = test_h20)

#Deep learning 

deep <- h2o.deeplearning(y = Y, training_frame = train_h20, nfolds = 10,
                   keep_cross_validation_predictions = TRUE, 
                   stopping_rounds = 50, stopping_tolerance = 0, stopping_metric = "RMSE", seed = 123, fold_assignment = "Modulo")

h2o.performance(deep, newdata = test_h20)

# Creating the ensembling tree

ensemble <- h2o.stackedEnsemble(y= Y, training_frame = train_h20, base_models = list(rf, gbm), 
                                metalearner_algorithm = "drf")

h2o.performance(ensemble, newdata = test_h20)

####### Interpreting the ensemble tree ##########

#Model specific understanding

vip(glm, method = "model")
vip(rf, method = "model")
vip(gbm, method = "model")
vip(deep, method = "model")

# Setting up local predictions to explian

preds <- predict(ensemble, train_h20) %>% as.vector()

# Selecting three sets of feature values - maximum spending, minimum spending and average spending

max_spend <- as.data.frame(train_h20)[which.max(preds), ] %>% select(-AmountSpent)
min_spend <- as.data.frame(train_h20)[which.min(preds), ] %>% select(-AmountSpent)

mean_pred <- filter(data.frame(preds), preds >= (mean(preds)-7), preds <= (mean(preds)+7))

mean_spend <- as.data.frame(train_h20)[which(preds == as.numeric(mean_pred[1,])), ] %>% select(-AmountSpent)

#Model agnostic

# We will need 3 things to carry out Model Agnostic Interpretations

# 1) data frame with the features (not the prediction harget)

features <- as.data.frame(train_h20) %>% select(-AmountSpent)

# 2) Create a vector with the actual responses
response <- as.data.frame(train_h20) %>% pull(AmountSpent)

# 3) Create custom predict function that returns the predicted values as a vector

pred <- function(object, newdata){
  results <- as.vector(h2o.predict(object, as.h2o(newdata)))
  return(results)
}


# Now we will create model agnostic objects for Interpretation

# for IML package
components_iml <- Predictor$new(model = ensemble, data = features, y = response, predict.function = pred)

# for DALEX package
components_dalex <- DALEX::explain(model = ensemble, data = features, y = response, predict_function = pred)

##### Permutation Based Feature Importance ##########

# using the vip library

vip(ensemble,
    train = as.data.frame(train_h20),
    method = "permute",
    target = "AmountSpent",
    metric = "RMSE",
    nsim = 3,
    sample_frac = 0.5,
    pred_wrapper = pred)


##### Partial Dependence ####

# using the pdp library

pdp_pred <- function(object, newdata)  {
  results <- mean(as.vector(h2o.predict(object, as.h2o(newdata))))
  return(results)
}

p1 <- partial(ensemble, pred.var = "Salary", train = as.data.frame(train_h20), pred.fun = pdp_pred)

autoplot(p1, rug = TRUE, train = as.data.frame(train_h20))              

p2 <- partial(ensemble, pred.var = "Catalogs", train = as.data.frame(train_h20), pred.fun = pdp_pred)

autoplot(p2, rug = TRUE, train = as.data.frame(train_h20))

########## Individual Conditional Expectation (ICE) #######

#using the pdp library

partial(ensemble, pred.var = "Salary", train = as.data.frame(train_h20), pred.fun = pred, plot = TRUE, center = TRUE)

partial(ensemble, pred.var = "Catalogs", train = as.data.frame(train_h20), pred.fun = pred, plot = TRUE, center = TRUE)

######### Feature Interactions ##########

#using the iml library

#one way overall interactions

interact <- Interaction$new(components_iml)

plot(interact)

#paired interactions

interact_2way <- Interaction$new(components_iml, feature = "Salary")

plot(interact_2way)


##### Surrogate Model ###########

# Using the iml package

tree <- TreeSurrogate$new(components_iml, maxdepth = 3)

plot(tree)


###### Local Interpretable Model Agnostic Explanations #########

# Using the LIME Package

components_lime <- lime(x = features, model = ensemble, n_bins = 10)

lime_explanation <- lime::explain(x = rbind(max_spend, mean_spend, min_spend),
                                  explainer = components_lime,
                                  n_features = 9)

glimpse(lime_explanation)

plot_features(lime_explanation, ncol = 1)

##### Shapley Values #######

# Using IML package

shapley <- Shapley$new(components_iml, x.interest = max_spend)
plot(shapley)

shapley$explain(x.interest = mean_spend)
plot(shapley)

shapley$explain(x.interest = min_spend)
plot(shapley)


########## Break Down Approach ###########

# Using DALEX library

high_break <- predict_parts(components_dalex, new_observation = max_spend, type = "break_down")
plot(high_break)

mid_break <- predict_parts(components_dalex, new_observation = mean_spend, type = "break_down")
plot(mid_break)

min_break <- predict_parts(components_dalex, new_observation = min_spend, type = "break_down")
plot(min_break)
