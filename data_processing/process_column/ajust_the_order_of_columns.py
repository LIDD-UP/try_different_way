#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: ajust_the_order_of_columns.py
@time: 2018/7/31
"""
import pandas as pd
train = pd.read_csv('./month_6_1.csv')
test = pd.read_csv('./test_data_6_1.csv')

train_justify = train[['province','longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
test_justify = test[['province','longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]

train_justify = train_justify.dropna()
test_justify = test_justify.dropna()
train_justify[['buildingTypeId']] = train_justify[['buildingTypeId']].astype(str)
test_justify[['buildingTypeId']] = test_justify[['buildingTypeId']].astype(str)
train_justify = train_justify.loc[0:999,:]
print(train_justify.shape)
print(test_justify.shape)

train_justify.to_csv('./month_6_1_try.csv',index=False)
test_justify.to_csv('./test_data_6_1_try.csv',index=False)

