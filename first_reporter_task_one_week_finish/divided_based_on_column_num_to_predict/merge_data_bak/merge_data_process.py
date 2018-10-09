# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_data_process.py
@time: 2018/9/26
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
import missingno as msno
import  matplotlib.pyplot as plt

merge1_data = pd.read_csv('./merge_data_auto_ml_test_listing_17.csv')
merge2_data = pd.read_csv('./merge_data_auto_ml_test_listing_18.csv')
origin_data = pd.read_csv('./merge_data_auto_ml_origin.csv')


print(merge1_data.shape)
print(merge2_data.shape)

# merge = pd.concat((merge2_data,merge1_data),axis=0)




# print(merge.shape)
# print(origin_data.shape)
# print(merge.head())

# print(mean_absolute_error(merge['daysOnMarket'],merge['predictions']))

# print(merge['daysOnMarket'].describe())
# print(merge['predictions'].describe())


def get_value(merge1_data, merge2_data):
    new_prediciton = []
    merge1_data_list = list(merge1_data['predictions'])
    merge2_data_list = list(merge2_data['predictions'])
    for i in range(len(merge2_data)):
        new_prediciton.append(0.1*merge1_data_list[i] + 0.9*merge2_data_list[i])

    return new_prediciton


new_prediciton = get_value(merge1_data, merge2_data)
print(mean_absolute_error(merge2_data['daysOnMarket'],new_prediciton))