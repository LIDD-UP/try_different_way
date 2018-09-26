# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_data_process.py
@time: 2018/9/26
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
import missingno as msno
import  matplotlib.pyplot as plt

merge1_data = pd.read_csv('./merge_data_auto_ml.csv')
merge2_data = pd.read_csv('./merge_data_auto_ml_inverse.csv')


print(merge1_data.shape)
print(merge2_data.shape)

merge = pd.concat((merge2_data,merge1_data),axis=0)

print(merge.shape)

print(mean_absolute_error(merge['daysOnMarket'],merge['predictions']))