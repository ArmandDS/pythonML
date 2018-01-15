# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:04:41 2016

@author: Usuario
"""

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


## Load Data

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
house_train = pd.read_csv('kc_house_train_data.csv',sep=',', header = 0, dtype=dtype_dict)
house_test = pd.read_csv('kc_house_test_data.csv',sep=',', header = 0, dtype=dtype_dict)

house_train['bedrooms_squared']= house_train['bedrooms']*house_train['bedrooms']
house_train['bed_bath_rooms']= house_train['bedrooms']*house_train['bathrooms']
house_train['lat_plus_long']= house_train['lat'] +house_train['long']
house_train['log_sqft_living']= house_train['sqft_living'].apply(math.log)

house_test['bedrooms_squared']= house_test['bedrooms']*house_test['bedrooms']
house_test['bed_bath_rooms']= house_test['bedrooms']*house_test['bathrooms']
house_test['lat_plus_long']= house_test['lat'] +house_test['long']
house_test['log_sqft_living']= house_test['sqft_living'].apply(math.log)

##Mean Bedroom
a= house_test['bedrooms_squared'].mean()
# Mean Bed Bath
b= house_test['bed_bath_rooms'].mean()
# Mean Lat + long
c= house_test['lat_plus_long'].mean()
#Mean log
d= house_test['log_sqft_living'].mean()


###Models

features1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat','long']
features2 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms']
features3 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


## Builde model 1

x1 = house_train[features1]
y1 = house_train['price']
lm1 = linear_model.LinearRegression()
model1 = lm1.fit(x1,y1)

## Build model 2
x2 = house_train[features2]
y2 = house_train['price']
lm2 = linear_model.LinearRegression()
model2 = lm2.fit(x2,y2)

## Build model 3
x3 = house_train[features3]
y3 = house_train['price']
lm3 = linear_model.LinearRegression()
model3 = lm3.fit(x3,y3)


batccoef = lm1.coef_[2]

## for 2

batccoef2 = lm2.coef_[2]
## Rss on Training
## RSS 1

rss1 = np.mean((lm1.predict(x1) - y1) ** 2)


## RSS 2

rss2 = np.mean((lm2.predict(x2) - y2) ** 2)

## RSS 2

rss3 = np.mean((lm3.predict(x3) - y3) ** 2)



##RSS on test

## RSS 1

x1t = house_test[features1]
y1t = house_test['price']

x2t = house_test[features2]
y2t = house_test['price']


x3t = house_test[features3]
y3t = house_test['price']


rss1t = np.mean((lm1.predict(x1t) - y1t) ** 2)


## RSS 2

rss2t = np.mean((lm2.predict(x2t) - y2t) ** 2)

## RSS 2

rss3t = np.mean((lm3.predict(x3t) - y3t) ** 2)

