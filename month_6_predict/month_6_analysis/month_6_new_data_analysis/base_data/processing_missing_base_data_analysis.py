# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: processing_missing_base_data_analysis.py
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

data = pd.read_csv('./processing_missing_base.csv')
print(data.head())
print(data.shape)
# print(data.dtypes)
# print(data[['style','lotFront', 'garageSpaces']])
# print(data['garageSpaces'].mode())
print(data['ownershiptype'].mode())
print(data['style'].mode())
data['garageSpaces'] = data['garageSpaces'].fillna(data['garageSpaces'].mode()) # 此处的mode有问题：


# msno.bar(data)
# plt.show()

'''
数据存在的问题：
    1：对于longitude数是负数的问题；需要将数据原本longitude的数据加上绝对值之后；再行进行
    2：对于实际业务中对多类别的空的情况实际是毫无意义的情况下，缺失值的填充问题，是drop还是填充众数，   
    3：
    
'''
'''
该数据存在的问题：
    1：style：由于缺失情况对实际毫无意义，是drop掉还是，用众数填充，（先drop掉）看情况；
    2：lotFront: lot可以解释为地段的意思：地段前面，可以用0填充
    3：由于脚本存在逻辑上的缺陷，导致roomiWidth 未填充完全，
    4:garageSpaces:应该用0进行填充；车库应该用0填充；
'''