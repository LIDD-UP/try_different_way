#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data.py
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
import missingno

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

data_train_6 = pd.read_csv('../input/month_6_1.csv')
data_test_6 = pd.read_csv('../input/test_data_6_1.csv')
# data_train_6_process = pd.read_csv('../input/test_data_6_processing.csv')
data_train_6_process = pd.read_csv('../input/month_6_processing.csv')


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

# sns.pairplot(data_train_6_process)
# plt.show()

# 对于latitude 小于40的数据也不要了；
# price 和 daysonmarket做log变换；
# 统计一下longitude 的数据大于-125 的数量；不是特别大就去掉；


# 看一下bedrooms 的分布
# sns.barplot(data_train_6_process['bedrooms'])

# plt.hist(data_train_6_process['bedrooms'])
# plt.show()

# 有一种方式看成三个高斯分布；

# 统计一下处理过后bedrooms数据；
# print(data_train_6_process['bedrooms'].value_counts())
'''
3.0    8750
2.0    5785
4.0    5430
1.0    2052
5.0    1823
6.0     428
0.0     159
7.0      82
'''
# 处理之前的bedrooms数据
# print(data_train_6['bedrooms'].value_counts())
# print(data_test_6['bedrooms'].value_counts())

# 分成三类数据感觉不太合适，对于bedrooms这种类型的数据应该做one_hot编码比较合适；
# 但是在编码之前是否应该只取12345 之类的数据；数据主要分布在这几个上，先只取这几个把，把数据弄得干净一点；


# 进行处理：把小于40纬度的剔除；
# 把price 和价格，做log变换；
# 对bedrooms只取12345 的数据，其余的不要，做one_hot 编码

data_train_6_process = data_train_6_process[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
data_train_6_process = data_train_6_process[data_train_6_process.latitude>40]
data_train_6_process['price'] = np.log1p(data_train_6_process['price'])
data_train_6_process['daysOnMarket'] = np.log1p(data_train_6_process['daysOnMarket'])
data_train_6_process = data_train_6_process[data_train_6_process.bedrooms<=5]
data_train_6_process = data_train_6_process[data_train_6_process.bedrooms>= 1]
# print(data_train_6_process.dtypes)

data_train_6_process['bedrooms'] = data_train_6_process['bedrooms'].astype(str)
data_train_6_process['buildingTypeId'] = data_train_6_process['buildingTypeId'].astype(float)
data_train_6_process['buildingTypeId'] = data_train_6_process['buildingTypeId'].astype(str)

# print(len(data_train_6_process[data_train_6_process.longitude<-125]))
data_train_6_process = data_train_6_process[data_train_6_process.longitude>-125]
# data_train_6_process = pd.get_dummies(data_train_6_process)
print(data_train_6_process.head())
data_train_6_process.to_csv('./final_process_train_6_dnn.csv',index=False)
# print(data_train_6_process.dtypes)
# print(data_train_6_process.shape)
# print(data_train_6_process.isna().sum())
# print(data_train_6_process.tail())

# print(data_train_6_process.head())

# 显示图
# sns.pairplot(data_train_6_process)
# missingno.bar(data_train_6_process)
# plt.show()


# 获取数据
# train = data_train_6_process[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]
# test = data_test_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]
# train_label = data_train_6_process['daysOnMarket']










