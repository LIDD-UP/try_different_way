#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: kaggle_price_predict.py
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
train = pd.read_csv('./kaggle_house_data/train.csv')
test = pd.read_csv('./kaggle_house_data/test.csv')

print(train.head())
print(test.head())
print(train.shape)
print(test.shape)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
print(all_data.head())
print(all_data.shape)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()

