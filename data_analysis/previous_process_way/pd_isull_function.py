#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: pd_isull_function.py
@time: 2018/7/12
"""
import pandas as pd

# 获取特征列为空的长度
def column_null_len(dataframe,column):
    column_is_null = pd.isnull(dataframe[column])
    column_is_null_true = column_is_null[column_is_null]
    coulumn_is_null_len = len(column_is_null_true)
    return coulumn_is_null_len

# 获取特征不为空的长度
def column_not_null_count(dataframe,column):
    column_is_null = pd.isnull(dataframe[column])
    column_is_null_true = column_is_null[column_is_null]
    column_is_null_len = len(column_is_null_true)
    column_not_null_len= len(dataframe[column])-column_is_null_len
    return column_not_null_len

