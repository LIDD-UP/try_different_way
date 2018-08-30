# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: missing_process.py
@time: 2018/8/29
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice
import  missingno as msno

data = pd.read_csv('processing_missing.csv')
print(data.dtypes)

print(data.head())
print(data['room2Width'].value_counts())
# msno.bar(data)
# plt.show()


# data = data[['priovince','city',]]
new_data = pd.DataFrame()
for column in data.columns:
    if len(data[pd.isnull(data[column])])/len(data[column]) <0.4:
        new_data[column] = data[column]

# msno.bar(new_data)
# plt.show()

print(new_data.shape)
new_data = new_data.dropna()
print(new_data.shape)
print(new_data.head())
print(new_data.dtypes)


