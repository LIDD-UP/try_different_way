# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: predict_analysis.py
@time: 2018/9/20
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error

# data = pd.read_csv('merge_data_auto_ml.csv')
# data = pd.read_csv('merge_data_auto_ml_xgboost.csv')
data = pd.read_csv('../input/treb_toronto_9.csv')



data_10 = []
data_20 = []
data_30 = []
data_more = []

# data = data.drop(columns=['index'])

test_column = 'projectDaysOnMarket'
# test_column = 'predictions'


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
0.3420238095238095
0.2695238095238095
0.07547619047619047
0.31297619047619046
28.07547619047619
'''

'''
my predictions
0.26134169884169883
0.2592905405405405
0.19063706563706564
0.288730694980695
24.81238204908187
'''
'''
0.3322876447876448
0.2548262548262548
0.16011100386100385
0.2527750965250965
22.444079292096685
'''


