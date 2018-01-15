# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 01:34:30 2016

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
import math   





def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1']= feature    
    #...
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
          # assign poly_dataframe[name] to be feature^power; use apply(*)
              name = 'power_' + str(power)
              #poly_dataframe[name]=feature**power
              poly_dataframe[name]=poly_dataframe['power_1'].apply(lambda x: x**power)
            #...
    return poly_dataframe
    
    
    


def print_15model(data):
    poly15_data = polynomial_dataframe(data['sqft_living'], 15)
    my_features = poly15_data.columns # get the name of the features
    poly15_data['price'] = data['price'] # add price to the data since it's the target
    model15 = LinearRegression()
    model15.fit(poly15_data[my_features], poly15_data['price'])
    print pd.Series(model15.coef_,index=my_features)
    plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
            poly15_data['power_1'], model15.predict(poly15_data[my_features]),'-')

    
    
    
    
    
    
    
    
    


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

#house_train = pd.read_csv('kc_house_train_data.csv',sep=',', header = 0, dtype=dtype_dict)

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'], axis=0)

poly1_data = polynomial_dataframe(sales['sqft_living'], 15)
features = poly1_data.columns # get the name of the features
x = poly1_data[features]
poly1_data['price'] = sales['price']


#feature = [ 'power_2']

y = poly1_data['price']
lm = LinearRegression()
lm.fit(x, y)

#matplotlib inline
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], lm.predict(poly1_data[features]),'-')







set1 = pd.read_csv('wk3_kc_house_set_1_data.csv')
set2 = pd.read_csv('wk3_kc_house_set_2_data.csv')
set3 = pd.read_csv('wk3_kc_house_set_3_data.csv')
set4 = pd.read_csv('wk3_kc_house_set_4_data.csv')




print_15model(set4)




test = pd.read_csv('wk3_kc_house_test_data.csv')
train = pd.read_csv('wk3_kc_house_train_data.csv')
valid = pd.read_csv('wk3_kc_house_valid_data.csv')
rss =[]
minimo = 0
for i in range(1,16):
    dta =polynomial_dataframe(train['sqft_living'], i)
    fet = dta.columns
    dta['price']= train['price']
    lm = LinearRegression()
    lm.fit(dta[fet], dta['price'])
    
    polyi_valid = polynomial_dataframe(valid['sqft_living'], i)
    pred_v = lm.predict(polyi_valid)
    rss_v = np.sum((pred_v - valid['price'])**2)
    
    #if rss_v<min[1]:
    #    min = (i, rss_v)
    polyi_test = polynomial_dataframe(test['sqft_living'], i)
    pred_t = lm.predict(polyi_test)
    rss_t = np.sum((pred_t - test['price'])**2)
    rss.append(rss_v)
    #rss.append(rss_t)
    if minimo == 0:
        minimo = rss_v
    
    if min(rss) < minimo:
        minimo = min(rss)
    
    print 'RSS for model %d- Validation : %f | Test : %f | #feat: %d' % (i, rss_v,rss_t,len(fet))
    
    
    
    
poly7_data = polynomial_dataframe(train['sqft_living'], 7)
my_features = poly7_data.columns # get the name of the features
poly7_data['price'] = train['price'] # add price to the data since it's the target
model7 = LinearRegression()
model7.fit(poly7_data[my_features], poly7_data['price'])
poly7_test = polynomial_dataframe(test['sqft_living'], 7)
pred = model7.predict(poly7_test)
rss = np.sum((pred - test['price'])**2)
print rss