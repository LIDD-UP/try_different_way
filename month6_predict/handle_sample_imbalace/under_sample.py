#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: under_sample.py
@time: 2018/8/9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

train_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_train.csv')
test_data = pd.read_csv('../standard_longitude_latitude/standard_log_lat_train.csv')

print(train_data.head())
print(test_data.head())

#采用下采样策略 对buildingtypeid进行处理：
print(train_data['buildingTypeId'].value_counts())

'''
 1:获取少样本的个数
 2:根据少样本的个数随机获取多样本的个数；获取
 3：获取多样本的下标，然后随机再这些下表中选择少样本个数的下标
 4：然后通过下标拿到少样本
 5：最后将这些样本组合起来；
 '''
# 获取少样本的个数
less_sample_len = len(train_data[train_data.buildingTypeId==2])
print(less_sample_len)

# 获取少样本的下标
less_sample_index_list = list(train_data[train_data.buildingTypeId==2].index)

multi_sample_index_list = list(train_data[train_data.buildingTypeId==1].index)
print(multi_sample_index_list)

random_multi_sample_index_list = sample(multi_sample_index_list,less_sample_len)
print(len(random_multi_sample_index_list))

# 合并下标
merge_index = less_sample_index_list + random_multi_sample_index_list
print(merge_index)

train_data_under_sample = train_data[train_data.buildingTypeId.index.isin(merge_index)]

print(train_data_under_sample.head())
print(train_data_under_sample['buildingTypeId'].value_counts())

train_data_under_sample.to_csv('under_sample_buildingTypeId.csv',index=False)

# 对bedrooms进行下采样
print(train_data_under_sample['bedrooms'].value_counts())

# 取出bedrooms最少样本的个数
bedrooms_less_sample_len = len(train_data_under_sample[train_data_under_sample.bedrooms==5.0])
print(bedrooms_less_sample_len)

# 取出最小样本的下标列表
bedrooms_less_sample_list = list(train_data_under_sample[train_data_under_sample.bedrooms==5.0].index)

# 取出其他种类样本的下标列表
bedrooms_1 =list(train_data_under_sample[train_data_under_sample.bedrooms==1.0].index)
bedrooms_2 =list(train_data_under_sample[train_data_under_sample.bedrooms==2.0].index)
bedrooms_3 =list(train_data_under_sample[train_data_under_sample.bedrooms==3.0].index)
bedrooms_4 =list(train_data_under_sample[train_data_under_sample.bedrooms==4.0].index)

# 进行随机选择：
random_bedrooms_1_list = sample(bedrooms_1,bedrooms_less_sample_len)
random_bedrooms_2_list = sample(bedrooms_2,bedrooms_less_sample_len)
random_bedrooms_3_list = sample(bedrooms_3,bedrooms_less_sample_len)
random_bedrooms_4_list = sample(bedrooms_4,bedrooms_less_sample_len)

# 合并
bedrooms_merge = random_bedrooms_1_list + random_bedrooms_2_list +random_bedrooms_3_list +random_bedrooms_4_list + bedrooms_less_sample_list

train_data_under_sample_bedromms = train_data_under_sample[train_data_under_sample.bedrooms.index.isin(bedrooms_merge)]

print(train_data_under_sample_bedromms['bedrooms'].value_counts())

# train_data_under_sample_bedromms.to_csv('under_sample_train.csv',index=False)



