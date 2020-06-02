# Predicting Bike buyers 
Building a binary classification model based on Logistic Regression that predicts the number of bike buyers using classification in Machine Learning.

# Overview
This is a logistic regression model to identify correlations between the following 6 independent customer features and assign label ( or 0) predicting wether the customer will buy a bike or not

### Part 1: Data Preprocessing
##### 1. Importing the dataset and Feature Selection
Imported the pandas library to read the dataset. The given dataset is multivariate defined over several different attributes. Each attribute is an integer.

Plotted various numerical and categorical features to examine thier relationships with bike_buyers in provided dataset by making boxplots and barplots. 6 usefull features are then identified to be used further for prediction.

##### 2. Splitting the dataset into a training set and test set
The dataset was split using the test_train_split function imported from model_selection.

Out of total instances, 20% were splitted into test set and remaining 80% were kept to train the dataset called as X_train, y_train, X_test, y_test.
The categorical features are one hot encoded and the nemerical features are scaled.
    
### Part 2: Training and Inference
##### 1. Training the logistic regression model on the training set
Trained the logistic regression model to fit on the splitted X_train and y_train dataset.

##### 2. Predicting the test results
The trained classifier was used to predict the values in test set, and later on a entirely new dataset(AW_test)

### Part 3: Evaluating the model
##### 1. Making the confusion matrix
The confusion matrix was created for the test set to evaluate the model depicting the approx percentage of the correct predicted value comparing from the given test set values. 

##### 2. Computing the accuracy and reiteration
Went back and changed the features, thershold value for sigmoid function comparision, applied weights to 0 and 1 in model to increase precision, Recall and F1. Calculated the Accuracy (79% approx) in both test set and the new data.

