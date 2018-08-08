#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: standard_and_normalization.py
@time: 2018/7/20
"""
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
data =pd.read_csv('month_4_1.csv',header=0)
data =data.dropna()
data = data.drop(columns=['listingDate','province','city','address'])


# 标准化：
# x = np.array(data['price']).reshape(-1,1)
# std = StandardScaler()
#
# data['price'] = std.fit_transform(x)


# 归一化：
# x = np.array(data['price']).reshape(-1,1)
# minmax = MinMaxScaler()
#
# data['price'] = minmax.fit_transform(x)

# print(data['price'])
#
# print(data[['price','daysOnMarket']])
# print(data['buildingTypeId'].dtype)


# 对所有特征进行归一化：

minmax = MinMaxScaler()
data_price = np.array(data[['longitude', 'price', 'latitude', 'buildingTypeId']])
data[['longitude', 'price', 'latitude', 'buildingTypeId']] = minmax.fit_transform(data_price)
print(data['buildingTypeId'])