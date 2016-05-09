# Machine Learning Course Project
Elisa Peirano  
9 de mayo de 2016  



## Summary
In this report, the data set on Weight Lifting Exercise from http://groupware.les.inf.puc-rio.br/har is used in order to predict the manner in which the exercise was done based on accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

## Loading the data

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "training.csv", method = "curl")
training <- read.csv("training.csv")
dim(training)
```

```
## [1] 19622   160
```

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "testing.csv", method = "curl")
testing <- read.csv("testing.csv")
```
## Pre processing the data
In taking a look at the training data set, we can see many columns with most of the values NAs or simply empty. As those aren't good predictors, we proceed to deleting them.
Also, the first 7 columns are just informative so we won't be using them either for the prediction.

```r
train <- training[, !(names(training) %in% names(training)[1:7])]
train <- train[, colSums(is.na(train))==0]
lista <- c()
for (i in 1:ncol(train)){if(length(which(train[,i]==""))!=0){lista <- c(lista,names(train)[i])}}
train <- train[, !(names(train) %in% lista)]
```

As seen before, the training data set has over 19600 registries, so in order to be able to work with the data and provide different sets for training a model, we will split the training in 4 groups of about the same length.

```r
library(caret)
set.seed(1209)
part1 <- createDataPartition(y = train$classe, p = .25, list=FALSE)
ds1 <- train[part1,]
rem <- train[-part1,]
part2 <- createDataPartition(y=rem$classe, p=.33, list=FALSE)
ds2 <- rem[part2,]
rem <- rem[-part2,]
part3 <- createDataPartition(y=rem$classe, p=0.5, list=FALSE)
ds3 <- rem[part3,]
ds4 <- rem[-part3,]
```
Then for each of the groups we subset a training and testing set with a proportion of 70-30 percent.

```r
set.seed(1209)
inTrain <- createDataPartition(y=ds1$classe, p=.7, list=FALSE)
train1 <- ds1[inTrain,]
test1 <- ds1[-inTrain,]
inTrain <- createDataPartition(y=ds2$classe, p=.7, list=FALSE)
train2 <- ds2[inTrain,]
test2 <- ds2[-inTrain,]
inTrain <- createDataPartition(y=ds3$classe, p=.7, list=FALSE)
train3 <- ds3[inTrain,]
test3 <- ds3[-inTrain,]
inTrain <- createDataPartition(y=ds4$classe, p=.7, list=FALSE)
train4 <- ds4[inTrain,]
test4 <- ds4[-inTrain,]
```
## Building the prediction model
There are several ways and method to build a predicition model. In this case we will compare predictiong with trees ("rpart" method) and predictiong with random forests ("rf" method). These were chosen for the interaction between variables they use and because of ramdom forests being one of the top performing algorithms.

Fitting a "rpart" model for the first data set.

```r
set.seed(1209)
fit1 <- train(train1$classe ~ ., data = train1, method = "rpart")
pred1 <- predict(fit1, test1)
confusionMatrix(pred1, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 391 136 128 111  47
##          B   4  87  10  40  29
##          C  23  62 118  90  68
##          D   0   0   0   0   0
##          E   0   0   0   0 126
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4912         
##                  95% CI : (0.4653, 0.517)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3321         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9354  0.30526  0.46094   0.0000  0.46667
## Specificity            0.5989  0.92996  0.79984   1.0000  1.00000
## Pos Pred Value         0.4809  0.51176  0.32687      NaN  1.00000
## Neg Pred Value         0.9589  0.84769  0.87556   0.8361  0.89286
## Prevalence             0.2844  0.19388  0.17415   0.1639  0.18367
## Detection Rate         0.2660  0.05918  0.08027   0.0000  0.08571
## Detection Prevalence   0.5531  0.11565  0.24558   0.0000  0.08571
## Balanced Accuracy      0.7671  0.61761  0.63039   0.5000  0.73333
```
As can be seen in the Confusion Matrix, the accuracy is very low: 0,4912 so there are many bad predictions.

Fitting a "rf" model for the first data set

```r
set.seed(1209)
fit2 <- train(train1$classe ~ ., data = train1, method = "rf")
pred2 <- predict(fit2, test1)
confusionMatrix(pred2, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 411  10   1   0   0
##          B   2 269   3   0   0
##          C   0   6 247   8   2
##          D   5   0   5 233   2
##          E   0   0   0   0 266
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9701        
##                  95% CI : (0.96, 0.9782)
##     No Information Rate : 0.2844        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9621        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9833   0.9439   0.9648   0.9668   0.9852
## Specificity            0.9895   0.9958   0.9868   0.9902   1.0000
## Pos Pred Value         0.9739   0.9818   0.9392   0.9510   1.0000
## Neg Pred Value         0.9933   0.9866   0.9925   0.9935   0.9967
## Prevalence             0.2844   0.1939   0.1741   0.1639   0.1837
## Detection Rate         0.2796   0.1830   0.1680   0.1585   0.1810
## Detection Prevalence   0.2871   0.1864   0.1789   0.1667   0.1810
## Balanced Accuracy      0.9864   0.9698   0.9758   0.9785   0.9926
```
In this case the accuracy is conseiderably better: 0.9721

Let's find out if it can be improved by using preProcessing and cross validation


```r
set.seed(1209)
fit22 <- train(train1$classe ~ ., data = train1, method = "rf", preProcess=c("center", "scale"), trControl = trainControl(method = "cv", number = 4))
pred2 <- predict(fit2, test1)
confusionMatrix(pred2, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 411  10   1   0   0
##          B   2 269   3   0   0
##          C   0   6 247   8   2
##          D   5   0   5 233   2
##          E   0   0   0   0 266
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9701        
##                  95% CI : (0.96, 0.9782)
##     No Information Rate : 0.2844        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9621        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9833   0.9439   0.9648   0.9668   0.9852
## Specificity            0.9895   0.9958   0.9868   0.9902   1.0000
## Pos Pred Value         0.9739   0.9818   0.9392   0.9510   1.0000
## Neg Pred Value         0.9933   0.9866   0.9925   0.9935   0.9967
## Prevalence             0.2844   0.1939   0.1741   0.1639   0.1837
## Detection Rate         0.2796   0.1830   0.1680   0.1585   0.1810
## Detection Prevalence   0.2871   0.1864   0.1789   0.1667   0.1810
## Balanced Accuracy      0.9864   0.9698   0.9758   0.9785   0.9926
```

The accuracy for the model with cross validation and pre-processing actually drops a tiny bit. So, let's just use the model without them for the rest of the data sets

```r
set.seed(1209)
fitt2 <- train(train2$classe ~ ., data = train2, method = "rf")
pred3 <- predict(fitt2, test2)
confusionMatrix(pred3, test2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 408   7   1   0   0
##          B   3 270   7   1   2
##          C   3   5 244   1   1
##          D   0   0   2 236   1
##          E   0   0   0   0 263
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9766          
##                  95% CI : (0.9675, 0.9838)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9704          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9855   0.9574   0.9606   0.9916   0.9850
## Specificity            0.9923   0.9889   0.9917   0.9975   1.0000
## Pos Pred Value         0.9808   0.9541   0.9606   0.9874   1.0000
## Neg Pred Value         0.9942   0.9898   0.9917   0.9984   0.9966
## Prevalence             0.2845   0.1938   0.1746   0.1636   0.1835
## Detection Rate         0.2804   0.1856   0.1677   0.1622   0.1808
## Detection Prevalence   0.2859   0.1945   0.1746   0.1643   0.1808
## Balanced Accuracy      0.9889   0.9732   0.9762   0.9946   0.9925
```

```r
fitte2 <- train(train3$classe ~ ., data = train3, method = "rf")
pred4 <- predict(fitte2, test3)
confusionMatrix(pred4, test3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 412  10   0   0   0
##          B   2 268   4   0   0
##          C   2   8 249  11   4
##          D   4   0   5 230   0
##          E   0   0   0   1 267
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9655          
##                  95% CI : (0.9548, 0.9742)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9563          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9810   0.9371   0.9651   0.9504   0.9852
## Specificity            0.9905   0.9950   0.9795   0.9927   0.9992
## Pos Pred Value         0.9763   0.9781   0.9088   0.9623   0.9963
## Neg Pred Value         0.9924   0.9850   0.9925   0.9903   0.9967
## Prevalence             0.2844   0.1936   0.1747   0.1638   0.1835
## Detection Rate         0.2789   0.1814   0.1686   0.1557   0.1808
## Detection Prevalence   0.2857   0.1855   0.1855   0.1618   0.1814
## Balanced Accuracy      0.9857   0.9660   0.9723   0.9716   0.9922
```

```r
fitted2 <- train(train4$classe ~ ., data = train4, method = "rf")
pred5 <- predict(fitted2, test4)
confusionMatrix(pred5, test4$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 417  12   0   0   0
##          B   1 268  15   0   3
##          C   1   4 234   2   2
##          D   0   1   8 240   3
##          E   1   0   0   0 263
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9641         
##                  95% CI : (0.9533, 0.973)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9545         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9404   0.9105   0.9917   0.9705
## Specificity            0.9886   0.9840   0.9926   0.9903   0.9992
## Pos Pred Value         0.9720   0.9338   0.9630   0.9524   0.9962
## Neg Pred Value         0.9971   0.9857   0.9813   0.9984   0.9934
## Prevalence             0.2847   0.1932   0.1742   0.1641   0.1837
## Detection Rate         0.2827   0.1817   0.1586   0.1627   0.1783
## Detection Prevalence   0.2908   0.1946   0.1647   0.1708   0.1790
## Balanced Accuracy      0.9907   0.9622   0.9516   0.9910   0.9848
```

##Out of sample error
The out of sample error for each model are the follwing:  
*data set 1: 1-0.9721 = 0.0279  
*data set 2: 1-0.9766 = 0.0234  
*data set 3: 1-0.9655 = 0.0345  
*data set 4: 1-0.9641 = 0.0359  

The average out of sample error is: 

```
## [1] 0.0279
```

Seeing as the best accuracy is from the second data set, we use that model to predict the testing data set:


```r
predict(fitt2, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

##Conclusion
Considering a 97% accuracy as pretty good, it was only necessary to eliminate the variables from the data set that had too many NAs or no info at all.
Even though the random forest method can be quite slow, the great performace makes up for it.
