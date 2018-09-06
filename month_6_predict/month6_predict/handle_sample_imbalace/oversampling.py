#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: oversampling.py
@time: 2018/8/9
"""
'''
smote 样本生成

样本到其他样本的欧氏距离；
排序找到5个
xnew = x +rand(0,1)*(x-x);

需要一个库叫做i目标Learn.oner_smpling import smote

现载入smote算法：
oversample = SMOTE(random_state=0)

预测集不懂，需要动测试集；

os_features,os_labels = oversampler.fitsample(features_train,labels_train)


'''

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import os

# 获取项目根目录
input_data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/input/'
print(input_data_path)

# 获取数据得位置
month_6_train_path = input_data_path +'month_6_1.csv'
month_6_test_path = input_data_path + 'test_data_6_1.csv'

# 读取数据
data_train = pd.read_csv(month_6_train_path)
data_test = pd.read_csv(month_6_test_path)




# 取出这些数据；
train = data_train[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
# train= train.dropna()
test = data_test[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print(train.head())
print(test.head())

train = train.dropna()
train['buildingTypeId'] = train['buildingTypeId'].astype(int)
train['buildingTypeId'] = train['buildingTypeId'].astype(str)
print(train.dtypes)
print(train['buildingTypeId'].value_counts())
train_feature = train[['longitude', 'latitude', 'price',  'bedrooms','daysOnMarket']]
train_label = train['buildingTypeId']


print(train_feature.shape)
sampler = SMOTE(random_state=0)
train_feature,label = sampler.fit_sample(train_feature,train_label)

print(train_feature.shape)
train_feature = pd.DataFrame(train_feature)
print(train_feature.head)
print(len(label))
print(label)

# print(train_feature.loc[:,3].value_counts())


# 对bedrooms做过采样，

