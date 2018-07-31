#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: ajust_the_order_of_columns.py
@time: 2018/7/31
"""
import pandas as pd
train = pd.read_csv('./month_456_1.csv')
test = pd.read_csv('./test_data_6_1.csv')

train_justify = train[['province','city','address','longitude','latitude','price','buildingTypeId','listingDate','bedrooms','daysOnMarket']]
test_justigy = test[['province','city','address','longitude','latitude','price','buildingTypeId','listingDate','bedrooms','daysOnMarket']]


train_justify.to_csv('./new_month_456_1.csv',index=False)
test_justigy.to_csv('./new_test_data_1.csv',index=False)

