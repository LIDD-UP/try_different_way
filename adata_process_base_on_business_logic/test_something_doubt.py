#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_something_doubt.py
@time: 2018/8/16
"""
import pandas as pd

data = pd.read_csv('./month_6_1.csv',header=0)
data_null_len = len(data[pd.isnull(data['bedrooms'])])
print(data_null_len)

# 获取缺失值的另一种方式：直接让条件data[col] == data[col] 可能是缺失值不会相等造成的没有缺失的才会相等；
data_new = data.loc[data['bedrooms']==data['bedrooms']]


print(data_new.shape[0])
print(data.shape[0])

