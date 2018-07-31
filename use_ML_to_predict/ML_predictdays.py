#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: ML_predictdays.py
@time: 2018/7/31
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
pd.set_option('max_columns',200)
pd.set_option('display.width',1000)
train = pd.read_csv('./company_house_data/month_456_1.csv')
test = pd.read_csv('./company_house_data/test_data_6_1.csv')

# print(train.head())
# print(test.head())
# print(train.shape)
# print(test.shape)

# all_data = pd.concat((train.loc[:,'province':'SaleCondition'],
#                       test.loc[:,'MSSubClass':'SaleCondition']))
# print(all_data.head())
# print(all_data.shape)

prices = pd.DataFrame({"days":train["daysOnMarket"], "log(days + 1)":np.log1p(train["daysOnMarket"])})

# print(prices.head())
# print(prices.shape)
prices.hist()
plt.show()
'''
(125636, 11)
(860, 11)
   days  log(days + 1)
0      8        2.197225
1      8        2.197225
2     32        3.496508
3     82        4.418841
4     56        4.043051
(125636, 2)

'''
