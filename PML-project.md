# Practical Machine Learning Project
chaitra  


## Prediction on Exercise Performance
### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

### Project Goal

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We need to build a model, perform cross-validation and estimate out of sample error. We should then use the chosen model to predict 20 different test cases.

### Loading the required packages


```r
library(caret)
library(rpart)
library(randomForest)
library(rpart.plot)
```

### Getting and Cleaning the data

The train set and the test set data are downloaded from the given url to the local PML folder on the hard disk. 


```r
# setting the seed to ensure reproduceability
set.seed(35343)

# read the data into training and test sets respectively. 
# Care is taken to mark NAs and other special expressions as NAs.
training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

# filter out columns with NAs
training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]

# remove the unnecessary variables from the data sets
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
dim(training)
```

```
## [1] 19622    53
```

```r
dim(testing)
```

```
## [1] 20 53
```

### Machine Learning

The aim is to predict the 'classe' variable using Machine Learning algorithms. Since the original testing set does not contain the classe variable, we cannot validate the results on the testing data. So the training set itself is split into two subsets so that we can validate our model.


```r
inTrain <- createDataPartition(y=training$classe,p=0.75,list=FALSE)
mytraining <- training[inTrain,]
mytesting <- training[-inTrain,]
dim(mytraining)
```

```
## [1] 14718    53
```

```r
dim(mytesting)
```

```
## [1] 4904   53
```

### Prediction models : Classification Tree method

First, we use the decision tree method to build our model.

```r
modelTree <- rpart(classe ~.,data=mytraining,method="class")
prp(modelTree)
```

![](PML-project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
# cross validating on the sub sample set aside.
predictTree <- predict(modelTree,mytesting,type="class")
confusionMatrix(predictTree,mytesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1283  141   14   44   14
##          B   38  584   85   73   77
##          C   36  106  684  142  118
##          D   18   77   45  488   69
##          E   20   41   27   57  623
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7467          
##                  95% CI : (0.7343, 0.7589)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6789          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9197   0.6154   0.8000  0.60697   0.6915
## Specificity            0.9393   0.9310   0.9007  0.94902   0.9638
## Pos Pred Value         0.8576   0.6814   0.6298  0.70014   0.8112
## Neg Pred Value         0.9671   0.9098   0.9552  0.92489   0.9328
## Prevalence             0.2845   0.1935   0.1743  0.16395   0.1837
## Detection Rate         0.2616   0.1191   0.1395  0.09951   0.1270
## Detection Prevalence   0.3051   0.1748   0.2215  0.14213   0.1566
## Balanced Accuracy      0.9295   0.7732   0.8504  0.77799   0.8276
```

### Prediction models : Random Forest Method

Then we use the random forest method to build a second model

```r
modelRF <- randomForest(classe ~.,data=mytraining,method="class")
# cross validation
predictRF <- predict(modelRF,mytesting,type="class")
confusionMatrix(predictRF,mytesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    2    0    0    0
##          B    1  946    1    0    0
##          C    0    1  854    8    0
##          D    0    0    0  795    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9969         
##                  95% CI : (0.995, 0.9983)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9961         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   0.9988   0.9888   0.9989
## Specificity            0.9994   0.9995   0.9978   0.9998   0.9998
## Pos Pred Value         0.9986   0.9979   0.9896   0.9987   0.9989
## Neg Pred Value         0.9997   0.9992   0.9998   0.9978   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1741   0.1621   0.1835
## Detection Prevalence   0.2847   0.1933   0.1760   0.1623   0.1837
## Balanced Accuracy      0.9994   0.9982   0.9983   0.9943   0.9993
```

```r
plot(modelRF)
```

![](PML-project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

### Model selection

From the confusion matrices, we can see significant differences in kappa and accuracy values between the two models. The accuracy for Tree model is 74.7% and that for RF model is 99.7%. Consequently, the out of sample errors (1 - accuracy) for the two are 25.3% and 0.45% respectively. 

#### Hence we select the Random Forest method to build our final model.

### Final Prediction : Results

Using the final model on the original testing dataset, we predict the following results.


```r
predictFinal <- predict(modelRF,testing,type="class")
predictFinal
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


