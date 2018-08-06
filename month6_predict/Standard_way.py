#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: Standard_way.py
@time: 2018/8/6
"""
import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import seaborn as sns

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

data_train_6 = pd.read_csv('../input/month_6_1.csv')
data_test_6 = pd.read_csv('../input/test_data_6_1.csv')

# 去掉buildingTypeId 为空的情况避免再编码的时候出现na这一类

data_train_6 = data_train_6[pd.isna(data_train_6.buildingTypeId) != True]
data_test_6 = data_test_6[pd.isna(data_test_6.buildingTypeId) != True]


# 去掉省份城市地址，调整顺序
data_train_6 = data_train_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_train_6 shape:', data_train_6.shape)
data_test_6 = data_test_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_test_6 shape:', data_test_6.shape)

# 取出label：
data_train_6_label = data_train_6['daysOnMarket']
data_test_6_label = data_test_6['daysOnMarket']


sns.pairplot(data_train_6)
plt.show()


