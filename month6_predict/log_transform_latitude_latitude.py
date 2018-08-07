#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: log_transform_latitude_latitude.py
@time: 2018/8/7
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


train_data = pd.read_csv('./final_process_train_6_dnn.csv')
test_data = pd.read_csv('./final_process_test_6_dnn.csv')

train_data['longitude'] = np.log1p(train_data['longitude'])
train_data['latitude'] = np.log1p(train_data['latitude'])
test_data['longitude'] = np.log1p(test_data['longitude'])
test_data['latitude'] = np.log1p(test_data['latitude'])


print(train_data.head())
print(test_data.head())

# train_data =pd.get_dummies(train_data)
# test_data = pd.get_dummies(test_data)
#
# train_data.to_csv('./standard_longitude_latitude/standard_log_lat_train.csv',index=False)
# test_data.to_csv('./standard_longitude_latitude/standard_log_lat_test.csv',index=False)


