rm(list=ls())
setwd('C:/Users/DELL/Desktop/project 2')
getwd()
install.packages("fastDummies")
x = c("ggplot2", "DMwR", "randomForest", "e1071", "rpart", "scales", "fastDummies", "caTools")
lapply(x, require, character.only = TRUE)
data = read.csv('C:/Users/DELL/Desktop/project 2/project_2.csv', header = T)

#####missing value check#####
table(is.na(data))
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
missing_val

 

# Creating Dummy variables for non-binary variables and dropping irrelevant variables

data_proper = fastDummies::dummy_cols(data, select_columns = c("weekday", "season", "weathersit", "mnth"))
drop_col = c("weekday","weekday0", "season","season1", "weathersit","weathersit1", "mnth","mnth1", "instant", "dteday", "casual", "registered", "temp")
data_proper = data_proper[,!names(data_proper) %in% drop_col]
#data_nodummy = data[,!names(data) %in% drop_col]

#MAPE Evaluation Function
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}



#Divide the data into train and test
set.seed(1234)
train_index = sample(1:nrow(data_proper), 0.8 * nrow(data_proper))
train = data_proper[train_index,]
test = data_proper[-train_index,]
X_train = train[, !names(train) %in% c("cnt")]
y_train = train$cnt
X_test = test[, !names(test) %in% c("cnt")]
y_test = test$cnt


######### Regression Models ###########

######## Multiple linear regression

reg_lm = lm(cnt~., train)

# Predicting on test set
y_pred_lm = predict(reg_lm, X_test)

# Evaluating on test set
MAPE(y_test, y_pred_lm)


########### Decision Tree Regression
reg_dt = rpart(cnt~., train, control = rpart.control(minsplit = 1))

# Predicting on test set
y_pred_dt = predict(reg_dt, X_test)

# Evaluating on test set
MAPE(y_test, y_pred_dt)



############## Random Forest Regression
reg_rf = randomForest(x = X_train, y = y_train, ntree = 600)

# Predicting on test set
y_pred_rf = predict(reg_rf, X_test)

# Evaluating on test set
MAPE(y_test, y_pred_rf)



######## Random Forerst seems to be working best

