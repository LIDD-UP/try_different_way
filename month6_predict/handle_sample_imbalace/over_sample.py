#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: over_sample.py
@time: 2018/8/10
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from imblearn.over_sampling import SMOTE

over_sample_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_train.csv')
# test_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_test.csv')
# train_data['bedrooms'] = train_data['bedrooms'].astype(int)
# train_data['bedrooms'] = train_data['bedrooms'].astype(str)

# print(train_data.dtypes)
# # 将样本分为平衡特征与需要过采样的特征即label
# feature = train_data[['longitude','latitude','price','bedrooms','daysOnMarket']]
# label = train_data['buildingTypeId']
# print(feature.shape)
# print(label.shape)
#
#
# smoter = SMOTE(random_state=0)
# feature,label = smoter.fit_sample(feature,label)
#
# print(feature.shape)
# print(label.shape)
# feature = pd.DataFrame(feature,columns=['longitude','latitude','price','bedrooms','daysOnMarket'])
# label = pd.DataFrame(label,columns=['buildingTypeId'])
#
# over_sample_data = pd.concat((feature,label),axis=1)
# print(over_sample_data.shape)
# print(over_sample_data.head())
# over_sample_data = over_sample_data[over_sample_data.bedrooms.isin([1,2,3,4,5])] # 去掉bedrooms不为整数的点；
# print(over_sample_data['buildingTypeId'].value_counts())
#
#
# print(over_sample_data['bedrooms'].value_counts())
# # over_sample_data.to_csv('over_sample_buildingTypeId.csv',index=False)

# 对bedrooms 进行过采样
# 简单的过采样方法：
print(over_sample_data.head())
over_sample_data['bedrooms'] = over_sample_data['bedrooms'].astype(str)
feature_bedrooms = over_sample_data[['longitude','latitude','price','buildingTypeId','daysOnMarket']]
label_bedrooms = over_sample_data['bedrooms']
smoter2 = SMOTE()
feature_bedrooms,label_bedrooms = smoter2.fit_sample(feature_bedrooms,label_bedrooms)
print(feature_bedrooms.shape)
print(label_bedrooms.shape)
feature_bedrooms = pd.DataFrame(feature_bedrooms,columns=['longitude','latitude','price','buildingTypeId','daysOnMarket'])
label_bedrooms = pd.DataFrame(label_bedrooms,columns=['bedrooms'])

over_sample_data_bedrooms = pd.concat((feature_bedrooms,label_bedrooms),axis=1)
print(over_sample_data_bedrooms.shape)
print(over_sample_data_bedrooms.head())
# over_sample_data_bedrooms = over_sample_data_bedrooms[over_sample_data_bedrooms.bedrooms.isin([1,2,3,4,5])] # 去掉bedrooms不为整数的点；
print(over_sample_data_bedrooms['bedrooms'].value_counts())

over_sample_data_bedrooms.to_csv('over_sample_bedrooms.csv',index=False)






