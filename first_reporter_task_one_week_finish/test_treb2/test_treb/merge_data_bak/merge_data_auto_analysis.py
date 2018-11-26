# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_data_auto_analysis.py
@time: 2018/11/21
"""
import pandas as pd

data = pd.read_csv('./merge_data_auto_ml.csv')

data = data[abs(data.predictions-data.daysOnMarket)<10]
data = data[['predictions','daysOnMarket']]
print(len(data))
# print(data)