# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:41:46 2016

@author: Usuario
"""

import numpy as np
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from collections import Counter




def countw(text):
    words = text.split()
    worddict = {}

    for word in words:
        if word in worddict:
            worddict[word] += 1
        else:
            worddict[word] = 1
    return worddict
    
    
people = pd.read_csv("people_wiki.csv", header=0)

obama = people[people['name'] == 'Barack Obama']
people['wc']= people['text'].apply(countw)
people.head()

##Split the data
train, test = train_test_split(people, test_size=0.2, random_state=0)
x = Counter(obama['text'].item().split())
x.most_common
trax = TfidfVectorizer()
y= trax.fit(test['text'])


tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(train['wc'][0])


response = tfidf.fit_transform(obama['text'])
obama['tidf']= response
#print response
dictr = {}
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print feature_names[col], ' - ', response[0, col]
    dictr[feature_names[col]] = response[0, col]
    
    
obama = train[train['name'] == 'Barack Obama']