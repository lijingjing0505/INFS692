# Helper packages
library(dplyr)        
library(keras)  
library(tfruns)       
library(tfestimators)
library(readr)

df = read.csv("radiomics_completedata.csv")

#splitting the data into training and testing
index<-createDataPartition(df$Failure.binary,p=0.7,list=F)


#Test labels in the Failure.binary column (column 2)
Train_Features <- data.matrix(df[index,-2])
Train_Labels <- df[index,2]
Test_Features <- data.matrix(df[-index,-2])
Test_Labels <- df[-index,2]

#converting the labels into categorical
to_categorical(as.numeric(Train_Labels))[,c(-1)] -> Train_Labels
to_categorical(as.numeric(Test_Labels))[,c(-1)] -> Test_Labels

#summary statistics
summary(Train_Labels)

#printing the structures of the dataset
str(Train_Features)

#converting the features into matrix
as.matrix(apply(Train_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Train_Features
as.matrix(apply(Test_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Test_Features

#building the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 256, activation = "sigmoid", input_shape = ncol(Train_Features)) %>%
  layer_dropout(rate = 0.25) %>% 
  
  layer_dense(units = 128, activation = "sigmoid") %>%
  layer_dropout(rate = 0.25) %>% 
  
  layer_dense(units = 128, activation = "sigmoid") %>%
  layer_dropout(rate = 0.25) %>% 
  
  layer_dense(units = 64, activation = "sigmoid") %>%
  layer_dropout(rate = 0.25) %>%
  
  layer_dense(units = 2, activation = "softmax")

summary(model)


#backpropagating slide 15
compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))

#compiling the model slide 33
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

#train the model with epoch=10, batch size=128 and validation split=0.15
history <- model %>% 
  fit(Train_Features,Train_Labels, epochs = 10, batch_size = 128, validation_split = 0.15)

plot(history)

#evaluating using test datasets
model %>% evaluate(Test_Features,Test_Labels)


#model prediction
model %>%
  predict_classes(Test_Feature)