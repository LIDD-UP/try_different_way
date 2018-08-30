# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: processing_missing_base_base_data_analysis.py
@time: 2018/8/30
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

data = pd.read_csv('./processing_missing_base_base.csv')
print(data.head())
print(data.shape)

msno.bar(data)
plt.show()

'''
分析缺失值情况可以看出一个问题，对于roomLength的填充情况并不好,
逻辑情况有问题；除了roomi为条件之外，本身如果缺失
'''
