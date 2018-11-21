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
data = pd.read_csv('../input/treb_toronto_10.csv')



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

def compute_ratio2(data):
    test_column = 'predictions'
    data_len = len(data)
    print(len(data[abs(data.predictions - data.daysOnMarket) <= 10])/data_len)
    print(len(data[abs(data.predictions - data.daysOnMarket).all() > 10 and abs(
                data.predictions - data.daysOnMarket).all() <= 20])/data_len)
    print(len(data[abs(data.predictions - data.daysOnMarket) > 20 and abs(
                data.predictions - data.daysOnMarket) <= 30])/data_len)
    print(len(data[abs(data.predictions - data.daysOnMarket) > 30])/data_len)
    print(mean_absolute_error(data[test_column], data['daysOnMarket']))

# if __name__ == '__main__':
#     # compute_ratio2(data)





'''
9月份情况；
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
10月份的情况
0.36163372231483687
0.29328411526553944
0.09085496546796856
0.25422719695165513
27.666706358656825
'''


