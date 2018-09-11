# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: predict_data_process.py
@time: 2018/9/11
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import missingno as msno
import matplotlib.pyplot as plt

data = pd.read_csv('./predict_data.csv')

# msno.bar(data)
# plt.show()

print(data.shape)
data = data.dropna()
print(data.shape)


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

# 处理bedrooms(eval)
data = process_bedrooms(data)
print(data.head(100))
data = data.drop()

data.to_csv('predict_data_dropna.csv',index=False)


