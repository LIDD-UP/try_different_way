# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_quantile_based_buckets.py
@time: 2018/9/12
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice
import  missingno as msno
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')
print(train_data.head(100))
print(test_data.head())
print(train_data.shape)
print(test_data.shape)
train_data = train_data.dropna()

train_data = train_data[['province','postalCode','city',
                         'longitude','latitude',
                         'propertyType',
                         'tradeTypeId',
                         'listingDate',
                         'buildingTypeId',
                         'bedrooms',
                         'bathroomTotal',
                         'delislingDate',
                         'daysOnMarket',
                         ]]

# print(train_data.dtypes)
# standard1 = StandardScaler()
# train_data = standard1.fit_transform(train_data)
# print(train_data.head())

for column in train_data.columns:
    if train_data[column].dtype != 'object' and column!= 'daysOnMarket' and column!= 'tradeTypeId':
        train_data[column] = StandardScaler().fit_transform(np.array(train_data[column]).reshape(-1,1))


print(train_data.head())