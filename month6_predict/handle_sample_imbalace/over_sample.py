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

train_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_train.csv')
test_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_test.csv')

# 将样本分为平衡特征与需要过采样的特征即label
feature = train_data[['longitude','latitude','price','bedrooms','daysOnMarket']]
label = train_data['buildingTypeId']
print(feature.shape)
print(label.shape)


smoter = SMOTE(random_state=0)
feature,label = smoter.fit_sample(feature,label)

print(feature.shape)
print(label.shape)
feature = pd.DataFrame(feature,columns=['longitude','latitude','price','bedrooms','daysOnMarket'])
label = pd.DataFrame(label,columns=['buildingTypeId'])

over_sample_data = pd.concat((feature,label),axis=1)
print(over_sample_data.shape)
print(over_sample_data.head())
print(over_sample_data['buildingTypeId'].value_counts())

# 对bedrooms 进行过采样
print(over_sample_data['bedrooms'].value_counts())


