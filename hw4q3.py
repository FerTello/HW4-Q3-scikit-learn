## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

######################################### Reading and Splitting the Data ###############################################
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100


# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=100, shuffle=True, train_size=0.7, test_size=None)


# ############################################### Linear Regression ###################################################
# TODO: Create a LinearRegression classifier and train it.

reg = LinearRegression().fit(x_train, y_train)


# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
print("Linear Regression")

y_pred_train = reg.predict(x_train).round()
reg_train_accu = (accuracy_score(y_train, y_pred_train)*100).round()
print("Training Accuracy: ", str(reg_train_accu))

y_pred_test = reg.predict(x_test).round()
reg_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
print("Testing Accuracy: ", str(reg_test_accu))


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX
print("MLP")
mlp = MLPClassifier(random_state=100).fit(x_train, y_train)


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

y_pred_train = mlp.predict(x_train)
mlp_train_accu = (accuracy_score(y_train, y_pred_train)*100).round()
print("Training Accuracy: ", str(mlp_train_accu))

y_pred_test = mlp.predict(x_test)
mlp_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
print("Testing Accuracy: ", str(mlp_test_accu))


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX
print("Random Forests Classifier")
rfc = RandomForestClassifier(random_state=100).fit(x_train, y_train)


# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.

y_pred_train = rfc.predict(x_train)
rfc_train_accu = (accuracy_score(y_train, y_pred_train)*100).round()
print("Training Accuracy: ", str(rfc_train_accu))

y_pred_test = rfc.predict(x_test)
rfc_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
print("Testing Accuracy: ", str(rfc_test_accu))



# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# print("Random Forest Hyper-parameters")
#
# param_grid = {
#     'max_depth': [80, 90, 100],
#     'n_estimators': [100, 200, 300]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_rfc_hp = GridSearchCV(rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1).fit(x_train, y_train)
# print(grid_rfc_hp.best_params_)
#
# y_pred_test = grid_rfc_hp.predict(x_test)
# grid_rfc_hp_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
# print("Testing Accuracy: ", str(grid_rfc_hp_test_accu))



# ############################################ Support Vector Machine ###################################################
print("SVM")
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# TODO: Create a SVC classifier and train it.
svm = SVC(random_state=100, gamma='auto').fit(x_train, y_train)


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_pred_train = svm.predict(x_train)
svm_train_accu = (accuracy_score(y_train, y_pred_train)*100).round()
print("Training Accuracy: ", str(svm_train_accu))

y_pred_test = svm.predict(x_test)
svm_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
print("Testing Accuracy: ", str(svm_test_accu))


# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
svm_hp = SVC(random_state=100, gamma='scale').fit(x_train, y_train)
print("SVM Hyper-parameters")
parameters = {'kernel': ['linear', 'rbf'], 'C': [0.01, 1, 100]}
grid_svm_hp = GridSearchCV(svm_hp, parameters, cv=10, n_jobs=-1, refit=True, verbose=1).fit(x_train, y_train)
# print(grid_svm_hp.best_params_)
# print(grid_svm_hp.best_score_)
# print(grid_svm_hp.cv_results_)

y_pred_test = grid_svm_hp.predict(x_test)
grid_svm_hp_test_accu = (accuracy_score(y_test, y_pred_test)*100).round()
print("Testing Accuracy: ", str(grid_svm_hp_test_accu))
