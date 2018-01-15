# -*- coding: utf-8 -*-
"""
Created on Sat May 21 02:58:31 2016

@author: Armand
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


## FUNCIONES

def word_count(string):
    #print type(string)
    string = str(string)
    my_string = string.lower().split()
    my_dict = {}
    for item in my_string:
        if item in my_dict:
            my_dict[item] += 1
        else:
            my_dict[item] = 1
    return (my_dict)

def awesome(dicc, word):
    #word = "wipes"
    if word in dicc:
        return dicc[word]
    else:
        return 0
    
    
data1 = pd.read_csv('amazon_baby.csv',sep=',', header = 0)
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
data1['word_count'] = data1['review'].apply(word_count)
for w in selected_words:
    data1[w] = data1['word_count'].apply(lambda dicc: awesome(dicc, w ))
##for item in selected_words:
  ##  data1[item]= data1['word_count'].apply(awesome,item)
 ##   print item
##
#print data1['word_count'][0]
data1.head()

for w in selected_words:
    print w+":"+str(sum(data1[w]))


print len(data1)


data1.head()


# Create a new sentiment analysis model using only the selected_words as features
data1 = data1[data1['rating'] != 3]
data1['sentiment'] = data1['rating'] >=4


##Split the data
train, test = train_test_split(data1, test_size=0.2, random_state=0)

## Training

feature = selected_words
x = train[feature]
y = train['sentiment']
logreg = linear_model.LogisticRegression()
logreg.fit(x,y)
selected_words_model = logreg.fit(x, y).coef_
xt = test[feature]
z = logreg.predict(xt)

## get the accuracy
acc = logreg.score(x,y)


## Examine Model 
# diaper_champ_reviews with the sentiment model
# -------------------------------------------
diaper_champ_reviews = data1[data1['name'] == 'Baby Trend Diaper Champ']
len(diaper_champ_reviews)

#x1 = train['word_count']
#y1 = train['sentiment']
#logregd = linear_model.LogisticRegression()
#logregd.fit(x1, y1)
#xt1 = test['word_count']
z1 = logreg.predict(diaper_champ_reviews[feature])
probs = logreg.predict_proba(diaper_champ_reviews[feature])