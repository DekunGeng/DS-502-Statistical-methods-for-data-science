# load packages

library(data.table)
library(corrplot)
library(Matrix)
library(xgboost)
require(randomForest)
require(caret)
require(dplyr)
require(ggplot2)
library(Metrics)
library(scales)

# read the data
train <- read.csv("train.csv")

## overview
dim(train)

## get list of character variables and numeric variables
cat_var <- names(train)[which(sapply(train, is.factor))]
numeric_var <- names(train)[which(sapply(train, is.numeric))]
##split train set
train_cat <- train[,cat_var]
train_cont <- train[,numeric_var]

# data clean
## view number of missing values per variable of character
missing_ct<-colSums(apply(X = train, MARGIN = 2, is.na))
missing_ct[which(missing_ct > 0)]
print(paste("There are ", length(missing_ct[which(missing_ct > 0)]), " variables have NAs."))
print(paste("Total number of NAs:", sum(missing_ct)))
## overview of character
colSums(sapply(train[,cat_var], is.na))[colSums(sapply(train[,cat_var], is.na)) != 0]
## overview of numericals
colSums(sapply(train[,numeric_var], is.na))[colSums(sapply(train[,numeric_var], is.na)) != 0] 
## Visualization for the missing data
plot_Missing <- function(data_in, title = NULL){
  temp_df <- as.data.frame(ifelse(is.na(data_in), 0, 1))
  temp_df <- temp_df[,order(colSums(temp_df))]
  data_temp <- expand.grid(list(x = 1:nrow(temp_df), y = colnames(temp_df)))
  data_temp$m <- as.vector(as.matrix(temp_df))
  data_temp <- data.frame(x = unlist(data_temp$x), y = unlist(data_temp$y), m = unlist(data_temp$m))
  ggplot(data_temp) + geom_tile(aes(x=x, y=y, fill=factor(m))) + scale_fill_manual(values=c("white", "#7EBBB9"), name="Missing\n(0=Yes, 1=No)") + theme_light() + ylab("") + xlab("") + ggtitle(title)
}
plot_Missing(train[,colSums(is.na(train)) > 0])

## summary
### The percentage of data missing in train.
sum(is.na(train)) / (nrow(train) *ncol(train))

## Check for duplicated rows.
cat("The number of duplicated rows are", nrow(train) - nrow(unique(train)))

## plot function
plotHist <- function(data_in, i) {
  data <- data.frame(x=data_in[[i]])
  p <- ggplot(data=data, aes(x=factor(x)),colour = "#7EBBB9") + stat_count() + xlab(colnames(data_in)[i]) + theme_light() + 
    theme(axis.text.x = element_text(angle = 90, hjust =1))
  return (p)
}

doPlots <- function(data_in, fun, ii, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(data_in=data_in, i=i)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
}


plotDen <- function(data_in, i){
  data <- data.frame(x=data_in[[i]], SalePrice = data_in$SalePrice)
  p <- ggplot(data= data,colour = "#7EBBB9") + geom_line(aes(x = x), stat = 'density', size = 1,alpha = 1.0) +
    xlab(paste0((colnames(data_in)[i]), '\n', 'Skewness: ',round(skewness(data_in[[i]], na.rm = TRUE), 2))) + theme_light() 
  return(p)
  
}

## Visualization
### char
doPlots(train_cat, fun = plotHist, ii = 1:16, ncol = 4)
doPlots(train_cat, fun = plotHist, ii = 17:32, ncol = 4)
doPlots(train_cat, fun = plotHist, ii = 33:43, ncol = 4)
### numeric
doPlots(train_cont, fun = plotDen, ii = 2:17, ncol = 4)
doPlots(train_cont, fun = plotDen, ii = 18:38, ncol = 4)

### find the relationship between neighborhoods and price
train %>% select(Neighborhood, SalePrice) %>% ggplot(aes(factor(Neighborhood), SalePrice)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 90, hjust =1)) + xlab('Neighborhoods')

## Explore correlation
### delete the na value
correlations <- cor(na.omit(train_cont[,-1]))

### correlations
row_indic <- apply(correlations, 1, function(x) sum(x > 0.3 | x < -0.3) > 1)

correlations<- correlations[row_indic ,row_indic ]
corrplot(correlations, method="square")

### plot scatter plot for variables that have high correlation
plotCorr <- function(data_in, i){
  data <- data.frame(x = data_in[[i]], SalePrice = data_in$SalePrice)
  p <- ggplot(data, aes(x = x, y = SalePrice)) + geom_point(shape = 1, na.rm = TRUE) + geom_smooth(method = lm ) + xlab(paste0(colnames(data_in)[i], '\n', 'R-Squared: ', round(cor(data_in[[i]], data$SalePrice, use = 'complete.obs'), 2))) + theme_light()
  return(suppressWarnings(p))
}

highcorr <- c(names(correlations[,'SalePrice'])[which(correlations[,'SalePrice'] > 0.5)], names(correlations[,'SalePrice'])[which(correlations[,'SalePrice'] < -0.2)])

data_corr <- train[,highcorr]

doPlots(data_corr, fun = plotCorr, ii = 1:12, ncol=4)

### salesprice

ggplot(train, aes(x=SalePrice)) + geom_histogram(col = 'white') + theme_light() +scale_x_continuous(labels = comma)

summary(train[,.(SalePrice)])

#### Normalize distribution
ggplot(train, aes(x=log(SalePrice+1))) + geom_histogram(col = 'white') + theme_light()

## Interactions based on correlation

for (x in numeric_var) {
  mean_value <- mean(train[[x]],na.rm = TRUE)
  train[[x]][is.na(train[[x]])] <- mean_value
}

for (name in colnames(correlations))
  print(c(name,names(correlations[,name])[which(correlations[,name] > 0.6)]))

train$livarea_qual <- train$OverallQual*train$GrLivArea #quality x living area
train$yearb_yearr_garyr <- train$YearBuilt*train$YearRemodAdd*train$GarageYrBlt 
train$bsmtFin_Bsmtfb <- train$BsmtFinSF1*train$BsmtFullBath
train$tBsmtSF_X1stFlrSf <- train$TotalBsmtSF*train$X1stFlrSF
train$X2ndFlrSF_livarea <- train$X2ndFlrSF*train$GrLivArea
train$X2ndFlrSF_halfb <- train$X2ndFlrSF*train$HalfBath
train$X2ndFlrSF_fullb <- train$X2ndFlrSF*train$FullBath
train$X2ndFlrSF_TotRmsAbvGr <- train$X2ndFlrSF*train$TotRmsAbvGr
train$livarea_FullBath <- train$GrLivArea*train$FullBath
train$livarea_TotRmsAbvGr <- train$GrLivArea*train$TotalBsmtSF
train$garageyrblt_cars <- train$GarageYrBlt*train$GarageCars



## factor to int

train[,cat_var] = sapply(train[,cat_var], as.numeric)

train[is.na(train)] = 0

# Prepare data

#outcome <- train$SalePrice

va <- sample(1:1460,730)
training <- train[-va,-1]
testing <- train[va,-1]


# Feature selection
## PCA
## Lasso

# Model
## linear regression
lm_model <- lm(SalePrice ~. ,training)
summary(lm_model)

par(mfrow=c(2,2))
plot(lm_model)

prediction <- predict(lm_model, testing, type="response")
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

rmse(model_output$log_SalePrice,model_output$log_prediction)


## LASSO
x = model.matrix(SalePrice~.,training)
y = training$SalePrice
grid=10^seq(10,-2,length=100)

set.seed (1)
tr=sample(1:nrow(x), nrow(x)/2)
test=(-tr)
y.test=y[test]

lasso.mod=glmnet(x[tr,],y[tr],alpha=1,lambda=grid)

set.seed (1)
cv.out=cv.glmnet(x[tr,],y[tr],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
rmse(log(lasso.pred),log(y.test))
lasso.coef = predict(lasso.mod,type = "coefficients",s=bestlam)[1:91,]
lasso.coef[lasso.coef != 0]

## RF
model_RF <- randomForest(SalePrice ~ ., data=training)


### Predict using the test set
prediction <- predict(model_RF, testing)
model_output <- cbind(testing, prediction)


model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

### Test with RMSE

rmse(model_output$log_SalePrice,model_output$log_prediction)

##Xgboost
#Assemble and format the data

training$log_SalePrice <- log(training$SalePrice)
testing$log_SalePrice <- log(testing$SalePrice)

#Create matrices from the data frames
trainData<- as.matrix(training, rownames.force=NA)
testData<- as.matrix(testing, rownames.force=NA)

#Turn the matrices into sparse matrices
train2 <- as(trainData, "sparseMatrix")
test2 <- as(testData, "sparseMatrix")

#####
#Cross Validate the model

vars <- c(2:37, 39:81) #choose the columns we want to use in the prediction matrix

trainD <- xgb.DMatrix(data = train2[,vars], label = train2[,"SalePrice"]) #Convert to xgb.DMatrix format

#Cross validate the model
cv.sparse <- xgb.cv(data = trainD,
                    nrounds = 600,
                    min_child_weight = 0,
                    max_depth = 10,
                    eta = 0.02,
                    subsample = .7,
                    colsample_bytree = .7,
                    booster = "gbtree",
                    eval_metric = "rmse",
                    verbose = TRUE,
                    print_every_n = 50,
                    nfold = 4,
                    nthread = 2,
                    objective="reg:linear")

#Train the model

#Choose the parameters for the model
param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.02,
              eval_metric = "rmse",
              objective="reg:linear")


#Train the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = trainD,
            nrounds = 600,
            watchlist = list(train = trainD),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

testD <- xgb.DMatrix(data = test2[,vars])
#Column names must match the inputs EXACTLY
prediction <- predict(bstSparse, testD) #Make the prediction based on the half of the training data set aside

#Put testing prediction and test dataset all together
test3 <- as.data.frame(as.matrix(test2))
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "prediction"
model_output <- cbind(test3, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE

rmse(model_output$log_SalePrice,model_output$log_prediction)

