#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_get_dummies.py
@time: 2018/7/31
"""
import pandas as pd


train = pd.read_csv('./company_house_data/month456_1_no_date.csv')
test = pd.read_csv('./company_house_data/test_data_6_1_no_date.csv')
y = pd.get_dummies(train,sparse=True)
print(y.shape)
# x= pd.get_dummies(test)
# print(x)