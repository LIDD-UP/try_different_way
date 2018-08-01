#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_pandas_function.py
@time: 2018/7/31
"""
import pandas as pd
train = pd.read_csv('./company_house_data/month_6_1_try.csv')
test = pd.read_csv('./company_house_data/test_data_6_1_try.csv')

numeric_feats = train.dtypes[train.dtypes != 'object'].index
# 还可以通过keys来取得；
# 也就是说他有一个索引关于行的索引，本身是一个Series
print(type(numeric_feats))
print(numeric_feats)
# print(list(numeric_feats.keys()))


