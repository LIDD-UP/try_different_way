# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_pandas_invert_to_dict.py
@time: 2018/10/22
"""
import pandas as pd

data = pd.read_csv('./input/month_6_1.csv')
print(data.head())
data_dict = dict(data)
# print(data_dict)
print(data_dict.keys())

print(type(data_dict['province']))
# for key,value in