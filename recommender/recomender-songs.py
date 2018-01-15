# -*- coding: utf-8 -*-

"""
Created on Fri Jun 03 22:23:20 2016

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



song = pd.read_csv("song_data.csv", header=0)
len(song)

kw = song[ song['artist']=='Kanye West']
print len(kw['user_id'].unique())


ff = song[ song['artist']=='Foo Fighters']
print len(ff['user_id'].unique())


ts = song[ song['artist']=='Taylor Swift']
print len(ts['user_id'].unique())



lg = song[ song['artist']=='Lady GaGa']
print len(lg['user_id'].unique())




kw = song[ song['artist']=='Kanye West']
##song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})

##SUMAR Y AGRUPAR POR CANCION Y ALBUM 
gs = kw.groupby(['song', 'title'])['listen_count'].count().sort_values(ascending= False)



##FF

##SUMAR Y AGRUPAR POR CANCION Y ALBUM 
gsff = ff.groupby(['song', 'title'])['listen_count'].count().sort_values(ascending= False)

## TS

gsts = ts.groupby(['song', 'title'])['listen_count'].count().sort_values(ascending= False)


##LG
gslg = lg.groupby(['song', 'title'])['listen_count'].count().sort_values(ascending= False)




## Group by artit

GA = song.groupby(['artist'])['listen_count'].count().sort_values(ascending= False)





train, test = train_test_split(song, test_size=0.2, random_state=0)

print len(test['user_id'].unique())