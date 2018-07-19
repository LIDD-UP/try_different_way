#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: feature_preceessing.py
@time: 2018/7/3
"""

#目标使用estimator自带的特征工程部分进行特征的处理（one_hot)编码方式

#使用pandas读取文件：

import tensorflow as tf
import pandas as pd
import os
import numpy as np

dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator\\house_info.csv'
test_filename = '\\use_estimator\\test_house_info.csv'
# 加载训练数据
data = pd.read_csv(dirname+train_filename,header=1,usecols=[4,5,6,10,11,14] ,names=['longitude','latitude','price','tradeTypeName','expectedDealPrice','daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
# print(data[['longitude']]) #他是一个二位的数组，需要有两个【】这个；
data = data.dropna(axis=0)
example = data[['longitude','latitude','price','tradeTypeName']]
label = data[['daysOnMarket']]



# 加载测试数据
data_test = pd.read_csv(dirname+test_filename,header=1,usecols=[4, 5, 6, 9,11, 14] ,names=['longitude','latitude','price','tradeTypeName','expectedDealPrice','daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
data_test = data_test.dropna(axis=0)
example_test = data_test[['longitude', 'latitude', 'price','tradeTypeName']]
label_test = data_test[['daysOnMarket']]

# print(data_test.to_dict(orient='list')) #把pandas转化成字典；

data_test_new = data_test.to_dict(orient='list')






# print(example,label_test)


import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder






# # 用bucktized_column编码：
# print(type(data_test))
#
# print(type(data_test['tradeTypeName']))
# result = np.ndarray(data_test)
# print(result)
#
# tradeTypeName_data_test = data_test
# tradeTypeName_column = feature_column.numeric_column('tradeTypeName')
# bucket_tradeTypeName = feature_column.bucketized_column(tradeTypeName_column,[0,1])
# tradeTypeName_bucket_tensor = feature_column.input_layer(tradeTypeName_data_test, [bucket_tradeTypeName])
# with tf.Session() as session:
#     print(session.run([tradeTypeName_bucket_tensor]))






#用feature_column.categorical_column_with_vocabulary_list方法行不通，相当于把每一个值进行了，标签编码并不是one_hot编码；
# 特征处理
tradeTypeName_data = data_test_new
builder = _LazyBuilder(tradeTypeName_data)
tradeTypeName_column = feature_column.categorical_column_with_vocabulary_list(
    'tradeTypeName', ['Sale', 'Lease'],dtype=tf.string, default_value=-1,
)

color_column_identy = feature_column.indicator_column(tradeTypeName_column)

print(type(tradeTypeName_data))
print(type(color_column_identy))
color_dense_tensor = feature_column.input_layer(tradeTypeName_data, [color_column_identy])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    print('use input_layer' + '_' * 40)
    print(session.run([color_dense_tensor]))