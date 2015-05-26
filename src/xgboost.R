require(xgboost)
require(caret)

set.seed(2000)

print("Preparing Data")

train <- read.csv('../Input/train.csv',header=TRUE,stringsAsFactors = F)
test <- read.csv('../Input/test.csv',header=TRUE,stringsAsFactors = F)
weather <- read.csv('../Input/weather.csv',header=TRUE,stringsAsFactors = F)

#Shuffle
train <- train[sample(nrow(train)),]

y = train$WnvPresent

train$WnvPresent     <- NULL
test$Id              <- NULL
train$NumMosquitos    <- NULL

trainlength = nrow(train)

x = rbind(train, test)

x$Year <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[1]))
x$Month <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[2]))
x$Week <- as.numeric(strftime(x$Date, format="%W"))

x$Latitude <- NULL
x$Longitude <- NULL
x$Date <- NULL
x$AddressNumberAndStreet <- NULL
x$Street <- NULL
x$AddressAccuracy <- NULL

dmy = dummyVars(" ~ . ", x)

x <- data.frame(predict(dmy, newdata = x))

headers = colnames(x)

x = as.matrix(x)

print(paste("scale_pos_weight=", sum(y==0) / sum(y==1)))

print("Training the model")

param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "nthread" = 16,
              "eta" = .02,
              "max_depth" = 10,
              "lambda_bias" = 0,
              "gamma" = .8,
              "min_child_weight" = 3,
              "subsample" = .6,
              "colsample_bytree" = .45,
              "scale_pos_weight" = sum(y==0) / sum(y==1),
              "base_score" = "0.8")

nround = 700
bst = xgboost(param=param, data = x[1:trainlength,], label = y, nrounds=nround, verbose = 2)

importance_matrix <- xgb.importance(headers, model = bst)
importance_matrix <- subset(importance_matrix, Gain > 0.01)
xgb.plot.importance(importance_matrix)

print("Making prediction")
pred = predict(bst, x[(nrow(train)+1):nrow(x),])
summary(pred)
pred = matrix(pred,1,length(pred))
pred = t(pred)

print("Storing Output")
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred), pred)
names(pred) = c('Id', 'WnvPresent')
write.csv(pred, file="../Output/nile-river-boost.csv", quote=FALSE,row.names=FALSE)
