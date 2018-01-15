# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import sklearn
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats




###Funcion Basica Regression

def simple_linear_regression(input_feature, output):
    x = sum(input_feature)
    y = sum(output)
    N = len(input_feature)
    xy = sum(input_feature*output)
    xx = sum(input_feature*input_feature)
    num = xy  - ((x*y)/N)
    den = xx - ((x*x)/N)
    slope = num/den
    intercept = y/N -(slope*x/N)
    
    return(intercept, slope)
    
    
    
##Funcion predictora
def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + slope*input_feature
    return(predicted_output)


## Funcio Error RSS
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    RSS = sum((output-(intercept + slope*input_feature))**2)
    return(RSS)


## Inverse Regresion
def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (-intercept + output)/slope
    return(estimated_input)

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
house_train = pd.read_csv('kc_house_train_data.csv',sep=',', header = 0, dtype=dtype_dict)

inputt= house_train['sqft_living']
bedom = house_train['bedrooms']
output = house_train['price']
a, b = simple_linear_regression(inputt, output)
print (a, b)
pre = get_regression_predictions(2650,a,b)
print pre
rss = get_residual_sum_of_squares(inputt,output,a,b)
print rss
inver = inverse_regression_predictions(800000, a,b)
print inver



a2, b2 = simple_linear_regression(bedom, output)
print(a2, b2)


## Modelo en test Data
house_test = pd.read_csv('kc_house_test_data.csv',sep=',', header = 0, dtype=dtype_dict)

itest = house_test['sqft_living']
otest = house_test['price']
bedt  = house_test['bedrooms']

rss = get_residual_sum_of_squares(itest,otest,a,b)
print rss

rss = get_residual_sum_of_squares(bedt,otest,a,b)
print rss