# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_train_data_process.py
@time: 2018/11/20
"""
import pandas as pd


data = pd.read_csv('./merge_train_data.csv')
data_orgin = data

# print(data.head())
print(data.shape)
data = data[abs(data.trainPrediction-data.daysOnMarket)<60]
# print(data.head())
print(data.shape)
data.to_csv('../input/treb_toronto_3to8_1.csv')
data_orgin.to_csv('./orgin_data.csv')
