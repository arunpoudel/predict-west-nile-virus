require(xgboost)

print("Preparing Data")

train <- read.csv('../Input/train.csv',header=TRUE,stringsAsFactors = F)
test <- read.csv('../Input/test.csv',header=TRUE,stringsAsFactors = F)

#Shuffle
train <- train[sample(nrow(train)),]

y = train$WnvPresent

train$WnvPresent     <- NULL
test$Id              <- NULL
train$NumMosquitos    <- NULL

trainlength = nrow(train)

x = rbind(train, test)

#x$Year <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[1]))
x$Month <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[2]))
x$Week <- as.numeric(strftime(x$Date, format="%W"))

x$month 			 <- x$Month
#x$year               <- x$Year
x$restuans           <- x$Species == 'CULEX RESTUANS'
x$pipiens            <- x$Species == 'CULEX PIPIENS'
x$both               <- x$Species == 'CULEX PIPIENS/RESTUANS'
x$territans          <- x$Species == 'CULEX TERRITANS'
x$latitude           <- x$Latitude
x$longitude          <- x$Longitude
x$block              <- x$Block

x$Month <- NULL
#x$Year <- NULL
x$Date <- NULL
x$NumMosquitos <- NULL
x$Species <- NULL
x$Latitude <- NULL
x$Longitude <- NULL
x$Address <- NULL
x$Block <- NULL
x$AddressNumberAndStreet <- NULL
x$Trap <- NULL
x$NumMosquitos <- NULL
x$Street <- NULL

print(x[1:20,])

x = x[,-1]

x = as.matrix(x)

print("Training the model")

param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "nthread" = 16,
              "eta" = .025,
              "max_depth" = 10,
              "lambda_bias" = 0,
              "gamma" = .8,
              "min_child_weight" = 3,
              "subsample" = .9,
              "colsample_bytree" = .45,
              "scale_pos_weight" = sum(y==0) / sum(y==1))

nround = 200
bst = xgboost(param=param, data = x[1:trainlength,], label = y, nrounds=nround, verbose = 2)

print("Making prediction")
pred = predict(bst, x[(nrow(train)+1):nrow(x),])
pred = matrix(pred,1,length(pred))
pred = t(pred)

print("Storing Output")
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred), pred)
names(pred) = c('Id', 'WnvPresent')
write.csv(pred, file="../Output/nile-river.csv", quote=FALSE,row.names=FALSE)