#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: house_info_2018_process.py
@time: 2018/7/18
"""
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# dirname = os.path.dirname(os.getcwd())
# train_filename = '\\use_estimator_new\\house_info_2018.csv'
# test_filename = '\\use_estimator_new\\test_house_info_2018.csv'
#
# # 加载训练数据
# data = pd.read_csv(dirname + train_filename, header=0, usecols=[1, 2, 3, 5, 6, 7, 8, 9, 10,11],
#                    names=['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId',
#                           'tradeTypeId', 'listingDate', 'daysOnMarket'])
#
# counts = pd.value_counts(data['tradeTypeId'])
# print(counts)
#
# new_data_1 = data[data.tradeTypeId==1]
# new_data_2 = data[data.tradeTypeId==2]
#
# print(new_data_1.shape)
# print(new_data_2.shape)
#
# new_data_1.to_csv('./house_info_2018_1.csv',index=False)
# new_data_2.to_csv('./house_info_2018_2.csv',index=False)

# X = new_data_1.ix[:,:-1]
# y = new_data_1.ix[:,-1:]
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

a = [1,2,3]
print(np.mean(a))