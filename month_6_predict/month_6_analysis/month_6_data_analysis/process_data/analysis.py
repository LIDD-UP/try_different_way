#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: analysis.py
@time: 2018/8/20
"""
import pandas as pd
import numpy
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import missingno as miss
import matplotlib.pyplot as plt





data = pd.read_csv('./month6.csv')

print(data.head())


print(data.describe())
print(len(list(data.columns)))

columns_set = []
for column in data.columns:
    if len(data[pd.notna(data[column]) ])>16000:
        columns_set.append(column)
data = data[columns_set]
print(data.shape)

print(data.head())
print(len(data['familyRoom']))

# print(data.dtypes)
# print(data['bedrooms'].value_counts())


def process_bedrooms(data):
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        if i != 'nan':
            list_month_process.append(eval(i))
        else:
            list_month_process.append(i)
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('float')
    return data
data = process_bedrooms(data)
print(data.head())
# print(data['bedrooms'].dtype)







# data = data.drop(columns='id')
# data.to_csv('./data_16000.csv',index=False)



