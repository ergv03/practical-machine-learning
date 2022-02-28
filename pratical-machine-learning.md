---
title: "Practical Machine Learning Project"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

## Data Preprocessing

Let's start by importing the necessary libraries and the data that will be used. The data is presented in the CSV format, and split into two files: one with the training data and another with the test data.

``` {r libraries, echo=FALSE}
library(lattice)
library(ggplot2)
library(caret)
library(rattle)
set.seed(1234)
```

``` {r data_loading}

trainData <- read.csv('/Users/egrochos/Projects/statistics-coursera/practical-machine-learning/pml-training.csv')
testData <- read.csv('/Users/egrochos/Projects/statistics-coursera/practical-machine-learning/pml-testing.csv')
dim(trainData)
dim(testData)

```

Before we run any modeling, we need to properly select which features to use. We do that by removing any columns that hold no predicting value (namely timestamps and window), columns that contains null values and finally columns that have close to zero variance. This way, we reduce the complexity of our model while minimizing the data noise. We run this process to both the training and test datasets, to make sure both are including the same columns and the prediction won't fail later.


``` {r data_prepoc}

colsRemove <- grepl("^X|timestamp|window", names(trainData))
trainData <- trainData[, !colsRemove]
testData <- testData[, !colsRemove]

cols_to_use <- apply(trainData, 2, function(x) sum(is.na(x)) == 0)
trainData <- trainData[,cols_to_use]
nvz <- nearZeroVar(trainData)
trainData <- trainData[, -nvz]
testData <- testData[,cols_to_use]
testData <- testData[, -nvz]

dim(trainData)
```

After the preprocessing above, we are left with 54 columns.

# Split training and validation subsets

Let's split the training dataset into training and validation sets.

``` {r data_split}
splitTrain <- createDataPartition(y=trainData$classe, p=0.7, list=F)
trainSubset <- trainData[splitTrain, ]
validateSubset <- trainData[-splitTrain,]
```
# Testing several models

Now that our data is properly preprocessed and we have the training and validation datasets, let's train different tree-based models and validate their performances against the validation dataset. We create a control variable that will handle the cross validation for us (with 3 folds).

``` {r control}
control <- trainControl(method="cv", number=3, verboseIter=F)
```
# Decision Tree

First we train a basic decision tree and store its results in a confusion matrix.

``` {r decisiontree}
tree_model <- train(classe~., data=trainSubset, method="rpart", trControl=control, tuneLength=2)
pred_trees <- predict(tree_model, validateSubset)
tree_cm <- confusionMatrix(pred_trees, factor(validateSubset$classe))
tree_cm
```

# Random Forest

Now we train a random forest.

``` {r rf}
rf_model <- train(classe~., data=trainSubset, method="rf", trControl=control, tuneLength=2)
pred_rf <- predict(rf_model, validateSubset)
rf_cm <- confusionMatrix(pred_rf, factor(validateSubset$classe))
rf_cm
```

# Gradient Boosting Decision Tree

And finally a gradient boost-based model.

``` {r gbm}
gbm_model <- train(classe~., data=trainSubset, method="gbm", trControl=control, tuneLength=2, verbose=F)
pred_gbm <- predict(gbm_model, validateSubset)
gbm_cm <- confusionMatrix(pred_gbm, factor(validateSubset$classe))
gbm_cm
```

# Models Scores

Using the confusion matrices generated above for each model, we get the following results (accuracy and error) when predicting the validation dataset:


| Model         |  Accuracy                       | Out of sample error                 |
|---------------|:-------------------------------:|------------------------------------:|
| Decision Trees| `r tree_cm$overall['Accuracy']` | `r 1 - tree_cm$overall['Accuracy']` |
| Random Forest | `r rf_cm$overall['Accuracy']`    | `r 1 - rf_cm$overall['Accuracy']`    |
| GBM           | `r gbm_cm$overall['Accuracy']`   | `r 1 - gbm_cm$overall['Accuracy']`   |

The model that performed the best (highest accuracy) on the validation dataset is the Random Forest, with an impressive 99% accuracy. Based on that, we use it to run a prediction on the test dataset:

``` {r pred_test}
testpred <- predict(rf_model, testData)
testpred
```
