# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: predict_analysis.py
@time: 2018/9/20
"""
import pandas as pd

data = pd.read_csv('merge_data_auto_ml.csv')

data_10 = []
data_20 = []
data_30 = []
data_more = []

data = data.drop(columns=['index'])


for i in range(len(data)):
    print(i)
    if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) <=10:
        data_10.append(i)
    if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) > 10 and abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) <=20:
        data_20.append(i)
    if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) > 20 and abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) <=30:
        data_30.append(i)
    if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) >30:
        data_more.append(i)

print(len(data_10)/len(data))
print(len(data_20)/len(data))
print(len(data_30)/len(data))
print(len(data_more)/len(data))
