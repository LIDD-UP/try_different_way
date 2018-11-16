#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data.py
@time: 2018/7/31
"""

import pandas as pd
import numpy as np
from scipy.stats import skew

month6 = pd.read_csv('../company_house_data/month_6_1.csv')
month456 = pd.read_csv('../company_house_data/month_456_1.csv')

print(month6.head())
# print(month456.head())

# month6_one_hot = pd.get_dummies(month456,sparse=True)
# print(month6_one_hot.head())

# 处理日期把日期拆分成为年月日三列：
def date_processing(_data):
    list_date = list(_data['listingDate'])
    list_break_together = []
    for data in list_date:
        list_break = data.split('/')
        list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together, columns=['year', 'month', 'day'], dtype='float32')
    return date_data_after_processing

# date_processing()
#log transform the target:
# train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = month6.dtypes[month6.dtypes != "object"].index
print('numeric_feats',numeric_feats)
print('-------------------------------')

skewed_feats = month6[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
print('skewed_feats',skewed_feats)
print('------------------------')
skewed_feats = skewed_feats[skewed_feats > 0.75]
print('skewed_feats',skewed_feats)
skewed_feats = skewed_feats.index
print('---------------------------------')
print('skewed_feats',skewed_feats)

month6[skewed_feats] = np.log1p(month6[skewed_feats])
print(month6.head())

