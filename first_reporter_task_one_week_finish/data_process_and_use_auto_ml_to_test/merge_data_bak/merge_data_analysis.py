# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_data_analysis.py
@time: 2018/9/21
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error

data_17 = pd.read_csv('./merge_data_auto_ml.csv')
data_18 = pd.read_csv('./merge_data_auto_ml.csv')

merge_data = pd.concat((data_17,data_18),axis=0)
print(mean_absolute_error(merge_data['daysOnMarket'],merge_data['predictions']))


