#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_processing.py
@time: 2018/7/5
"""
import os
import pandas as pd
import tensorflow as tf
#
# dirname = os.path.dirname(os.getcwd())
# train_filename = '\\use_estimator_new\\house_info.csv'
# test_filename = '\\use_estimator_new\\test_house_info.csv'
#
#
# data = pd.read_csv(dirname+train_filename,header=None)
# data = data.todict()
#
#
#
#
# # _CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [0.], [0.], [0.], [0.], [''], [0.], [''], [0.], [''], [''], [0.]]
# _CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [0.], [0.], [0.], [0.], [''], [0.], [''], [0.], [''], [''], [0.]]
#
#
# _CSV_COLUMNS=[
#     'province', 'city', 'address', 'postCode', 'longitude', 'latitude', 'price', 'buildingTypeId', 'buildingTypeName', 'tradeTypeId', 'tradeTypeName', 'expectedDealPrice', 'listingDate', 'delislingDate', 'daysOnMarket'
#
# ]
#
#
# province, city, address, postCode, longitude, latitude, price, buildingTypeId, buildingTypeName, tradeTypeId, tradeTypeName,expectedDealPrice, listingDate, delislingDate, daysOnMarket = tf.decode_csv(data, record_defaults=_CSV_COLUMN_DEFAULTS)
# # features = dict(zip(_CSV_COLUMNS, columns))
# # features.pop('postCode')
# # labels = features.pop('daysOnMarket')
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(columns))


# province, city, address, longitude, latitude, price, buildingTypeId, tradeTypeName, expectedDealPrice, daysOnMarket= tf.decode_csv(data, record_defaults=_CSV_COLUMN_DEFAULTS)
#
#
#
# example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeName', 'expectedDealPrice']]
# label = data[['daysOnMarket']]


#_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''],[''], [''],['']]

# _CSV_COLUMNS=[
#     'province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeName', 'expectedDealPrice', 'daysOnMarket'
# ]

import tensorflow as tf
import os
import numpy as np
import math


import tensorflow as tf
from sklearn import preprocessing
import numpy as np
from sklearn import linear_model


def read_data(file_queue):
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [''], [0.], [0.], [0.], [0.], [''],[0], [''], [0.], [''], [''], [0]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address, postCode, longitude,latitude, price, buildingTypeId, buildingTypeName, tradeTypeId, tradeTypeName, expectedDealPrice, listingDate, delislingDate, daysOnMarket = tf.decode_csv(value, record_defaults)
    feature = tf.stack([price, expectedDealPrice])
    return feature, daysOnMarket


def create_pipeline(filename,batch_size,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example1, dayOnMarket1 = read_data(file_queue) # example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    min_after_dequeue = 1000  # 出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    capacity = min_after_dequeue+batch_size#队列的长度
    # example_batch,label_batch= tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)#把队列的数据打乱了读取
    example_batch,daysOnMarket_batch= tf.train.batch([example1,dayOnMarket1],batch_size=batch_size,capacity=capacity) # 顺序读取
    return example_batch, daysOnMarket_batch

example_batch, daysOnMarket_batch = create_pipeline('house_info.csv',10000)



init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)

  coord = tf.train.Coordinator()#创建一个队列
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(100002):
    # Retrieve a single instance:
    example, label = sess.run([example_batch, daysOnMarket_batch])
    print('第%d批数据'%(i))
    print(example, label)

  coord.request_stop()
  coord.join(threads)



