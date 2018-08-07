#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: standard_longitude_latitude.py
@time: 2018/8/7
"""
'''
标准化longitude，latitude

'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


train_data = pd.read_csv('./final_process_train_6_dnn.csv')
test_data = pd.read_csv('./final_process_test_6_dnn.csv')


train_data['longitude'] = StandardScaler().fit_transform(np.array(train_data['longitude']).reshape(-1,1))
train_data['latitude'] = StandardScaler().fit_transform(np.array(train_data['latitude']).reshape(-1,1))
test_data['longitude'] = StandardScaler().fit_transform(np.array(test_data['longitude']).reshape(-1,1))
test_data['latitude'] = StandardScaler().fit_transform(np.array(test_data['latitude']).reshape(-1,1))

print(train_data.head())
print(test_data.head())

train_data =pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

train_data.to_csv('./standard_longitude_latitude/standard_log_lat_train.csv',index=False)
test_data.to_csv('./standard_longitude_latitude/standard_log_lat_test.csv',index=False)




