# -*- coding: utf-8 -*-
"""
Created on Sun May 22 02:13:14 2016

@author: Armand
"""


##Para borrar vairalbes comando reset
import sklearn
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math as math



def get_numpy_data(data, features, output):
    #data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’
    aux_matrix=data[features].as_matrix()
    ones = np.ones((len(data),1))
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = np.append(ones, aux_matrix, axis=1)
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = data[output].as_matrix() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)

def errors(output,predictions):
    errors=predictions-output
    return errors

def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


def feature_derivative(errors, feature):
    derivative = 2*np.dot(feature,errors)
    return(derivative)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    #Initital weights are converted to numpy array
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions=predict_outcome(feature_matrix,weights)
        # compute the errors as predictions - output:
        error=errors(output,predictions)
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            feature=feature_matrix[:, i]
            # compute the derivative for weight[i]:
            #predict=predict_outcome(feature,weights[i])
            #err=errors(output,predict)
            deriv=feature_derivative(error,feature)
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares=gradient_sum_squares+(deriv**2)
            # update the weight based on step size and derivative:
            weights[i]=weights[i] - np.dot(step_size,deriv)

        gradient_magnitude = math.sqrt(gradient_sum_squares)
        #stdout.write("\r%d" % int(gradient_magnitude))
        #stdout.flush()
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)



## Load Data

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
house_train = pd.read_csv('kc_house_train_data.csv',sep=',', header = 0, dtype=dtype_dict)
house_test = pd.read_csv('kc_house_test_data.csv',sep=',', header = 0, dtype=dtype_dict)



## Prepare data
simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(house_train, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size, tolerance)

(test_simple_feature_matrix ,test_output) =  get_numpy_data(house_test, simple_features, my_output)

predictedd = predict_outcome(test_simple_feature_matrix, simple_weights)




## RSS
rss1 = np.mean((predictedd - test_output) ** 2)

## For more than 1 varable
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(house_train, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

multi_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)


(test_multi_feature_matrix ,test_output) =  get_numpy_data(house_test, model_features, my_output)

predictedd2 = predict_outcome(test_multi_feature_matrix, multi_weights)



## RSS
rss2 = np.mean((predictedd2 - test_output) ** 2)