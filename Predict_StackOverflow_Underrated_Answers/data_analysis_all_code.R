library(data.table)
library(mlr)
library(dplyr)
library(caret)
library('RANN')
library(FSelector)

path<- "/Users/devadmin/Documents/886_final_project/output/csv4analysis/"
setwd(path)

data886 <- fread("csv120.csv", na.strings = c("", " ", "?", "NA", NA))
summarizeColumns(data886)

qid_lst <- data886$qid
aid_lst <- data886$aid
data_label <- data886$IsUnderrated
data886[, qid:=NULL]
data886[, aid:=NULL]
data886[, IsUnderrated:=NULL]

############################### Data Preprocessing ####################################
# remove zero variance data
variance_lst <- nearZeroVar(data886, saveMetrics = T)
zero_variance_list <- names(subset(data886, select = variance_lst$zeroVar==T))
zero_variance_list
data886[, (zero_variance_list):=NULL]
summarizeColumns(data886)

# check outliers
boxplot(data886$module_badname)
boxplot(data886$pylint_score)    # has more outliers
boxplot(data886$empty_percentage)
boxplot(data886$function_badname)

# find highly correlated data
ax <-findCorrelation(x = cor(data886), cutoff = 0.7) 
sort(ax)
data886 <- data886[, -ax, with=F]
summarizeColumns(data886)

# normalize features
normalized_data <- data.table(scale(data886))
summarizeColumns(normalized_data)

# all scale to [0,1]
data_scaling <- function(x){(x-min(x))/(max(x)-min(x))}
scaled_data <- data.table(sapply(data886, data_scaling))
summarizeColumns(scaled_data)

scaled_data[, IsUnderrated:=data_label]
scaled_data$IsUnderrated <- as.factor(scaled_data$IsUnderrated)

normalized_data[, IsUnderrated:=data_label]
normalized_data$IsUnderrated <- as.factor(normalized_data$IsUnderrated)

## in training data, 16 positive cases; in testing data, 7 positive cases


############################### Classification ####################################

idx <- createDataPartition(scaled_data$IsUnderrated, p=0.67, list=FALSE)

# DATASET 1 - scaled data set
scaled_train_data <- scaled_data[idx,]
scaled_test_data <- scaled_data[-idx,]
summary(scaled_train_data$IsUnderrated)
summary(scaled_test_data$IsUnderrated)

scaled_train_task <- makeClassifTask(data=data.frame(scaled_train_data), target = "IsUnderrated", positive = "Y")
scaled_test_task <- makeClassifTask(data=data.frame(scaled_test_data), target = "IsUnderrated", positive = "Y")

# DATASET 2 - normalized data set
normalized_train_data <- normalized_data[idx,]
normalized_test_data <- normalized_data[-idx,]
summary(normalized_train_data$IsUnderrated)
summary(normalized_test_data$IsUnderrated)

normalized_train_task <- makeClassifTask(data=data.frame(normalized_train_data), target = "IsUnderrated", positive = "Y")
normalized_test_task <- makeClassifTask(data=data.frame(normalized_test_data), target = "IsUnderrated", positive = "Y")


# Method 1 - Random Forest with all the data, and show feature importance
## with Scaled Dataset
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = scaled_train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=scaled_train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, scaled_test_task)
nb_prediction <- rfpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM     # all predicted as negative cases


## with Normalized Dataset
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = normalized_train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=normalized_train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, normalized_test_task)
nb_prediction <- rfpredict$data$response
dCM <- confusionMatrix(normalized_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM     # all predicted as negative cases


# Method 2 - SVM with all the data, and show feature importance
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = scaled_train_task, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, scaled_train_task)
svmpredict <- predict(svm_model, scaled_test_task)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   # all predicted as negative cases


# Method 3 - GBM with all the features  (DATASET IS TOO SMALL HERE, cannot get results) 
set.seed(410)
getParamSet("classif.gbm")
gbm_learner <- makeLearner("classif.gbm", predict.type = "response")
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_gbm <- makeResampleDesc("CV",iters = 3L)
gbm_param<- makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), 
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), 
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
gbm_tune <- tuneParams(learner = gbm_learner, task = scaled_train_task,resampling = cv_gbm,measures = acc,par.set = gbm_param,control = rancontrol)
gbm_tune$x
gbm_tune$y
final_gbm <- setHyperPars(learner = gbm_learner, par.vals = gbm_tune$x)
gbm_model <- mlr::train(final_gbm, scaled_train_task)
gbmpredict <- predict(gbm_model, scaled_test_task)
nb_prediction <- gbmpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM


# Method 4 - XGBoost with all the features
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)
xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = scaled_train_task, resampling = cv_xg, measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, scaled_train_task)
xgpredict <- predict(xgmodel, scaled_test_task)
nb_prediction <- xgpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   # all predicted as negative cases


# Method 5 - C50 with all the featues
getParamSet("classif.C50")
c50_learner <- makeLearner("classif.C50", predict.type = "response", par.vals = list(seed=410, noGlobalPruning=T, subset=T))
c50_param <- makeParamSet(
  makeIntegerParam("trials",lower = 10, upper = 100)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_c50 <- makeResampleDesc("CV",iters = 5L)
c50_tune <- tuneParams(learner = c50_learner, resampling = cv_c50, task = scaled_train_task, par.set = c50_param, control = rancontrol, measures = acc)
c50_tune$x
c50_tune$y
c50.tree <- setHyperPars(c50_learner, par.vals = c50_tune$x)
c50_model <- mlr::train(learner=c50.tree, task=scaled_train_task)
getLearnerModel(c50_model)
c50predict <- predict(c50_model, scaled_test_task)
nb_prediction <- c50predict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # all predicted as negative cases


library(ROSE)
bal_data_rose_train <- ROSE(IsUnderrated~., data = scaled_train_data, seed = 410)$data
# bal_data_rose_train <- ROSE(IsUnderrated~., data = normalized_train_data, seed = 410)$data
table(bal_data_rose_train$IsUnderrated)   # 143 N, 154 Y
train_task_rose <- makeClassifTask(data=data.frame(bal_data_rose_train), target = "IsUnderrated", positive = "Y")

# Method 6 - ROSE + SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = train_task_rose, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, train_task_rose)
svmpredict <- predict(svm_model, scaled_test_data)
# svmpredict <- predict(svm_model, normalized_test_data)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
# dCM <- confusionMatrix(normalized_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   
# scaled_data - balanced accuracy: 0.2828, Sensitivity : 0.90476, Specificity : 0.02913
# normalized_data - balanced accuracy: 0.02083, Sensitivity : 0, Specificity : 0.04167


# Method 7 - ROSE + Random Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = train_task_rose, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=train_task_rose)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, scaled_test_task)
nb_prediction <- rfpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.524306, Sensitivity : 1.000000, Specificity : 0.048611


# Method 8 - ROSE + GBM   (DAATSET TOO SMALL, cannot get results)
set.seed(410)
getParamSet("classif.gbm")
gbm_learner <- makeLearner("classif.gbm", predict.type = "response")
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_gbm <- makeResampleDesc("CV",iters = 5L)
gbm_param<- makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), 
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), 
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
gbm_tune <- tuneParams(learner = gbm_learner, task = train_task_rose,resampling = cv_gbm, measures = acc,par.set = gbm_param,control = rancontrol)
gbm_tune$x
gbm_tune$y
final_gbm <- setHyperPars(learner = gbm_learner, par.vals = gbm_tune$x)
gbm_model <- mlr::train(final_gbm, scaled_train_task)
gbmpredict <- predict(gbm_model, scaled_test_task)
nb_prediction <- gbmpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM


# Method 9 - XGBoost + ROSE
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)
xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = train_task_rose, resampling = cv_xg, measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, train_task_rose)
xgpredict <- predict(xgmodel, scaled_test_task)
nb_prediction <- xgpredict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   # Balanced Accuracy : 0.39628, Sensitivity : 0.75000, Specificity : 0.04255


# Method 10 - ROSE + C50
getParamSet("classif.C50")
c50_learner <- makeLearner("classif.C50", predict.type = "response", par.vals = list(seed=410, noGlobalPruning=T, subset=T))
c50_param <- makeParamSet(makeIntegerParam("trials",lower = 10, upper = 100))
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_c50 <- makeResampleDesc("CV",iters = 5L)
c50_tune <- tuneParams(learner = c50_learner, resampling = cv_c50, task = train_task_rose, par.set = c50_param, control = rancontrol, measures = acc)
c50_tune$x
c50_tune$y
c50.tree <- setHyperPars(c50_learner, par.vals = c50_tune$x)
c50_model <- mlr::train(learner=c50.tree, task=train_task_rose)
getLearnerModel(c50_model)
c50predict <- predict(c50_model, scaled_test_task)
nb_prediction <- c50predict$data$response
dCM <- confusionMatrix(scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.524306, Sensitivity : 1.000000, Specificity : 0.048611


# SMOTE has to remove near zero variance features
near_zero_variance_list <- c(2, 4, 7, 9, 10, 13, 14, 15, 22, 26)
smote_train_data <- scaled_train_data[, -near_zero_variance_list, with=F]
smote_test_data <- scaled_test_data[, -near_zero_variance_list, with=F]
smote_train_task <- makeClassifTask(data=data.frame(smote_train_data), target = "IsUnderrated", positive = "Y")
train_smote <- smote(smote_train_task, rate = 18, nn = 3)
smote_test_task <- makeClassifTask(data=data.frame(smote_test_data), target = "IsUnderrated", positive = "Y")

table(getTaskTargets(train_smote))    # 281 N, 288 Y
table(getTaskTargets(smote_test_task))

# Method 11 - SMOTE + Ramdom Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = train_smote, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=train_smote)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, smote_test_task)
nb_prediction <- rfpredict$data$response
nb_prediction
dCM <- confusionMatrix(smote_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy :0.475694, Sensitivity : 0, Specificity :  0.951389


# Method 12 - SMOTE + GBM
set.seed(410)
getParamSet("classif.gbm")
gbm_learner <- makeLearner("classif.gbm", predict.type = "response")
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_gbm <- makeResampleDesc("CV",iters = 5L)
gbm_param<- makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), 
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), 
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
gbm_tune <- tuneParams(learner = gbm_learner, task = train_smote, resampling = cv_gbm, measures = acc,par.set = gbm_param,control = rancontrol)
gbm_tune$x
gbm_tune$y
final_gbm <- setHyperPars(learner = gbm_learner, par.vals = gbm_tune$x)
gbm_model <- mlr::train(final_gbm, train_smote)
gbmpredict <- predict(gbm_model, smote_test_task)
nb_prediction <- gbmpredict$data$response
nb_prediction
dCM <- confusionMatrix(smote_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.645540, Sensitivity : 0.333333, Specificity :  0.957746


# SMOTE + XGBoost
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)
xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = train_smote, resampling = cv_xg, measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, train_smote)
xgpredict <- predict(xgmodel, smote_test_task)
nb_prediction <- xgpredict$data$response
nb_prediction
dCM <- confusionMatrix(smote_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   # Balanced Accuracy : 0.47552, Sensitivity : 0, Specificity : 0.95105


# SMOTE + C50
getParamSet("classif.C50")
c50_learner <- makeLearner("classif.C50", predict.type = "response", par.vals = list(seed=410, noGlobalPruning=T, subset=T))
c50_param <- makeParamSet(makeIntegerParam("trials",lower = 10, upper = 100))
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_c50 <- makeResampleDesc("CV",iters = 5L)
c50_tune <- tuneParams(learner = c50_learner, resampling = cv_c50, task = train_smote, par.set = c50_param, control = rancontrol, measures = acc)
c50_tune$x
c50_tune$y
c50.tree <- setHyperPars(c50_learner, par.vals = c50_tune$x)
c50_model <- mlr::train(learner=c50.tree, task=train_smote)
getLearnerModel(c50_model)
c50predict <- predict(c50_model, smote_test_task)
nb_prediction <- c50predict$data$response
dCM <- confusionMatrix(smote_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.47552, Sensitivity : 0, Specificity : 0.95105

## Summary: SMOTE works better with GBM than with XGBoost



# ROSE on all the data first, then with Random Forest
idx <- createDataPartition(scaled_data$IsUnderrated, p=0.67, list=FALSE)
rose_scaled_data <- ROSE(IsUnderrated~., data = scaled_data, seed = 410)$data
table(rose_scaled_data$IsUnderrated)   # 227 N, 215 Y

rose_scaled_train_data <- rose_scaled_data[idx,]
rose_scaled_test_data <- rose_scaled_data[-idx,]
summary(rose_scaled_train_data$IsUnderrated)   # N 159, Y 138
summary(rose_scaled_test_data$IsUnderrated)    # N 68, Y 77

rose_train_task <- makeClassifTask(data=data.frame(rose_scaled_train_data), target = "IsUnderrated", positive = "Y")
rose_test_task <- makeClassifTask(data=data.frame(rose_scaled_test_data), target = "IsUnderrated", positive = "Y")

# ROSE All + SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = rose_train_task, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, rose_train_task)
svmpredict <- predict(svm_model, rose_test_task)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(rose_scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.8697, Sensitivity : 0.8625, Specificity : 0.8769


# ROSE All + XGBoost
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)
xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = rose_train_task, resampling = cv_xg, measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, rose_train_task)
xgpredict <- predict(xgmodel, rose_test_task)
nb_prediction <- xgpredict$data$response
nb_prediction
dCM <- confusionMatrix(rose_scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM   # Balanced Accuracy : 1, Sensitivity : 1, Specificity : 1


# ROSE All + Random Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = rose_train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=rose_train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, rose_test_task)
nb_prediction <- rfpredict$data$response
nb_prediction
dCM <- confusionMatrix(rose_scaled_test_data$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 1, Sensitivity : 1.000000, Specificity : 1

## check generated feature importance
rf_feature_importance <- data.frame(rf_model$learner.model$importance)
feature_names <- data.frame(colnames(rose_scaled_train_data)[1:(length(rose_scaled_train_data)-1)])
colnames(feature_names)[1] <- 'Feature'
fi <- cbind(feature_names, rf_feature_importance$MeanDecreaseGini)
colnames(fi)[2] <- "MeanDecreaseGini"
setorder(fi, -"MeanDecreaseGini")
head(fi, n=15)

rf_selected_cols <- c(9, 26, 13, 22, 15, 10, 14, 12)
selected_train <- subset(rose_scaled_train_data, select = colnames(rose_scaled_train_data) %in% rf_selected_cols)
selected_train$IsUnderrated <-rose_scaled_train_data$IsUnderrated
selected_test <- subset(rose_scaled_test_data, select = colnames(rose_scaled_test_data) %in% rf_selected_cols)
selected_test$IsUnderrated <-rose_scaled_test_data$IsUnderrated

rf_train_task <- makeClassifTask(data=data.frame(selected_train), target = "IsUnderrated", positive = "Y")
rf_test_task <- makeClassifTask(data=data.frame(selected_test), target = "IsUnderrated", positive = "Y")

# Random Forest Selected features+ SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = rf_train_task, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, rf_train_task)
svmpredict <- predict(svm_model, rf_test_task)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(selected_test$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.6393, Sensitivity : 0.5476, Specificity : 0.5935


# Bi-varite Analysis - polt multiple plots together
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

summary(rose_scaled_data$IsUnderrated)   # 227 N, 215 Y


p1 <- scaled_data$module_documented[which(scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(p1), aes(x= p1, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
p2 <- scaled_data$module_documented[which(scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(p2), aes(x= p2, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

p1 <- rose_scaled_data$module_documented[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(p1), aes(x= p1, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
p2 <- rose_scaled_data$module_documented[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(p2), aes(x= p2, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$aswVerynegative[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$aswVerynegative[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$function_documented[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$function_documented[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$multi[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$multi[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$class_badname[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$class_badname[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$docstring_percentage[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(p1), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$docstring_percentage[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(p2), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- rose_scaled_data$warning_num[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- rose_scaled_data$warning_num[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)


# Boruta All-Relevant Feature Selection, with original scaled data
library(Boruta)
set.seed(410)
boruta_train <- Boruta(IsUnderrated~., data = scaled_data, doTrace = 2)
boruta_train
## plot feature importance
plot(boruta_train, xlab = "", xaxt = "n")
str(boruta_train)
summary(boruta_train$ImpHistory)
finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
                        function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(boruta_train$ImpHistory)
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)
## determine tentative features
new_boruta_train <- TentativeRoughFix(boruta_train)
new_boruta_train
plot(new_boruta_train, xlab = "", xaxt = "n")
finite_matrix <- lapply(1:ncol(new_boruta_train$ImpHistory), 
                        function(i) new_boruta_train$ImpHistory[is.finite(new_boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(new_boruta_train$ImpHistory) 
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(new_boruta_train$ImpHistory), cex.axis = 0.7)
feature_stats = attStats(new_boruta_train)
feature_stats
selected_cols <- getSelectedAttributes(new_boruta_train, withTentative = F)
selected_cols 

## generate training, testing data based on selected features
selected_train <- subset(scaled_train_data, select = colnames(scaled_train_data) %in% selected_cols)
selected_train[, IsUnderrated:=scaled_train_data$IsUnderrated]
selected_test <- subset(scaled_test_data, select = colnames(scaled_test_data) %in% selected_cols)
selected_test[, IsUnderrated:=scaled_test_data$IsUnderrated]

boruta_train_task <- makeClassifTask(data=data.frame(selected_train), target = "IsUnderrated", positive = "Y")
boruta_test_task <- makeClassifTask(data=data.frame(selected_test), target = "IsUnderrated", positive = "Y")

# Boruta + Random Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = boruta_train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=boruta_train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, boruta_test_task)
nb_prediction <- rfpredict$data$response
nb_prediction
dCM <- confusionMatrix(selected_test$IsUnderrated, nb_prediction, positive = "Y")
dCM  # all predicted as N


# Boruta + XGbOoost
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)
xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = boruta_train_task, resampling = cv_xg, measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, boruta_train_task)
xgpredict <- predict(xgmodel, boruta_test_task)
nb_prediction <- xgpredict$data$response
nb_prediction
dCM <- confusionMatrix(selected_test$IsUnderrated, nb_prediction, positive = "Y")
dCM    # all predicted as N


Y <- scaled_data$vote_gap[which(scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- scaled_data$vote_gap[which(scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- scaled_data$cmtPositive[which(scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- scaled_data$cmtPositive[which(scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- scaled_data$cmtNegative[which(scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- scaled_data$cmtNegative[which(scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

Y <- scaled_data$vote_percent[which(scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(Y), aes(x= Y, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
N<- scaled_data$vote_percent[which(scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(N), aes(x= N, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

## Summary: Boruta with original data, the selected features do not make a difference on prediction at all


# Boruta All-Relevant Feature Selection, with ROSE scaled data
library(Boruta)
set.seed(410)
boruta_train <- Boruta(IsUnderrated~., data = rose_scaled_data, doTrace = 2)
boruta_train
## plot feature importance
plot(boruta_train, xlab = "", xaxt = "n")
str(boruta_train)
summary(boruta_train$ImpHistory)
finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
                        function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(boruta_train$ImpHistory)
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)
## determine tentative features
new_boruta_train <- TentativeRoughFix(boruta_train)
new_boruta_train
plot(new_boruta_train, xlab = "", xaxt = "n")
finite_matrix <- lapply(1:ncol(new_boruta_train$ImpHistory), 
                        function(i) new_boruta_train$ImpHistory[is.finite(new_boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(new_boruta_train$ImpHistory) 
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(new_boruta_train$ImpHistory), cex.axis = 0.7)
feature_stats = attStats(new_boruta_train)
feature_stats
selected_cols <- getSelectedAttributes(new_boruta_train, withTentative = F)
selected_cols 

## Summary: Boruta get the same selected features as Random Forests, but ranking varies


## generate training, testing data based on selected features
selected_train <- subset(rose_scaled_train_data, select = colnames(rose_scaled_train_data) %in% selected_cols)
selected_train$IsUnderrated <-rose_scaled_train_data$IsUnderrated
selected_test <- subset(rose_scaled_test_data, select = colnames(rose_scaled_test_data) %in% selected_cols)
selected_test$IsUnderrated <-rose_scaled_test_data$IsUnderrated

boruta_train_task <- makeClassifTask(data=data.frame(selected_train), target = "IsUnderrated", positive = "Y")
boruta_test_task <- makeClassifTask(data=data.frame(selected_test), target = "IsUnderrated", positive = "Y")

# Boruta + Random Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = boruta_train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=boruta_train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, boruta_test_task)
nb_prediction <- rfpredict$data$response
nb_prediction
dCM <- confusionMatrix(selected_test$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 1, Sensitivity : 1, Specificity : 1


# Boruta + SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = boruta_train_task, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, boruta_train_task)
svmpredict <- predict(svm_model, boruta_test_task)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(selected_test$IsUnderrated, nb_prediction, positive = "Y")
dCM  # Balanced Accuracy : 0.9300, Sensitivity : 0.8929, Specificity : 0.9672




################################ Clustering ################################

summarizeColumns(normalized_data)
summarizeColumns(scaled_data)

normalized_data[, IsUnderrated:=NULL]
scaled_data[, IsUnderrated:=NULL]

# elbow method plot- find optimal 
# mydata <- normalized_data
mydata <- scaled_data
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# silhouette coefficient, check cluster similarity, higher the better
library (cluster)
library (vegan)
dis = vegdist(mydata)
res = pam(dis,10)
sil = silhouette(res$clustering,dis) # or use your cluster vector
plot(sil, border = NA)

# hierarchical cluster
d <- dist(scaled_data, method = "euclidean") # Euclidean distance matrix.
H.fit <- hclust(d, method="ward.D")   # Ward’s minimum variance criterion minimizes the total within-cluster variance
par(mar=rep(2,4))
plot(H.fit) # display dendogram
groups <- cutree(H.fit, k=5) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(H.fit, k=5, border="red") 

# Plot Y, N distribution in clusters
k.means.fit <- kmeans(normalized_data, 2)
k.means.fit$centers
k.means.fit$cluster
clust_label <- cbind(k.means.fit$cluster, data_label)
colnames(clust_label)[1] <- "Cluster"
colnames(clust_label)[2] <- "IsUnderrated"
clust_label <- data.frame(clust_label)
head(clust_label)
par(mfrow=c(2,1), mai=c(0.5, 0.5, 0.2, 0.2))    # mai here is to create lower chart
Y <- table(clust_label$Cluster[which(clust_label$IsUnderrated=='Y')])
barplot(Y, main="Y")
N <- table(clust_label$Cluster[which(clust_label$IsUnderrated=='N')])
barplot(N, main="N")


# Cluster Ensembling
library(clue)
d <- dist(scaled_data, method = "euclidean") # Euclidean distance matrix.
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty")
hclust_results <- lapply(hclust_methods, function(m) hclust(d, m))
names(hclust_results) <- hclust_methods
## Now create an ensemble from the results.
hens <- cl_ensemble(list = hclust_results)
hens
## Subscripting.
hens[1 : 3]
## Replication.
rep(hens, 3)
## Plotting.
plot(hens, main = names(hens))
## And continue to analyze the ensemble, e.g.
round(cl_dissimilarity(hens, method = "gamma"), 4)


# Clustering Y group and N group
scaled_data$IsUnderrated <- data_label
Y_group <- subset(scaled_data, IsUnderrated=="Y")
N_group <- subset(scaled_data, IsUnderrated=="N")

Y_group[, IsUnderrated:=NULL]
mydata <- Y_group
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

d <- dist(Y_group, method = "euclidean") # Euclidean distance matrix.
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty")
hclust_results <- lapply(hclust_methods, function(m) hclust(d, m))
names(hclust_results) <- hclust_methods
## Now create an ensemble from the results.
hens <- cl_ensemble(list = hclust_results)
hens
## Subscripting.
hens[1 : 3]
## Replication.
rep(hens, 3)
## Plotting.
plot(hens, main = names(hens))
