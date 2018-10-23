# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: predict_analysis.py
@time: 2018/9/20
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('merge_data_auto_ml.csv')
# data = pd.read_csv('merge_data_auto_ml_xgboost.csv')



data_10 = []
data_20 = []
data_30 = []
data_more = []

# data = data.drop(columns=['index'])

# test_column = 'projectDaysOnMarket'
test_column = 'predictions'


for i in range(len(data)):
    print(i)
    if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <=10:
        data_10.append(i)
    if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 10 and abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <=20:
        data_20.append(i)
    if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 20 and abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <=30:
        data_30.append(i)
    if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) >30:
        data_more.append(i)

print(len(data_10)/len(data))
print(len(data_20)/len(data))
print(len(data_30)/len(data))
print(len(data_more)/len(data))
print(mean_absolute_error(data[test_column],data['daysOnMarket']))

'''
0.25138180272108845
0.25690901360544216
0.1953656462585034
0.296343537414966


0.3410234899328859
0.26824664429530204
# 0.61
0.07560822147651007
0.31512164429530204
28.361052852348994
'''


