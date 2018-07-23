#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_read_data3.py
@time: 2018/7/23
"""
# columns = ['province', 'city', 'address','longitude','latitude', 'price', 'buildingTypeId',  'tradeTypeId',  'listingDate',  'daysOnMarket',bedrooms]

import tensorflow as tf
from sklearn import preprocessing
import numpy as np
from sklearn import linear_model
import numpy as np
np.set_printoptions(suppress=True)

def read_data(file_queue):
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [0.], [0.], [0.], [0.], [0.], [''],[0.],  [0.]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    columns = ['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
               'listingDate', 'daysOnMarket', 'bedrooms']

    columns_value = tf.decode_csv(value, record_defaults)
    return columns_value



def create_pipeline(filename,batch_size,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    columns_value = read_data(file_queue) # example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    min_after_dequeue = 1000  # 出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    capacity = min_after_dequeue+batch_size#队列的长度
    # example_batch,label_batch= tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)#把队列的数据打乱了读取
    columns_value_batch= tf.train.batch([columns_value],batch_size=batch_size,capacity=capacity) # 顺序读取
    return columns_value_batch

columns_value_batch = create_pipeline('month_4_1.csv',1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  # Start populating the filename queue.

  coord = tf.train.Coordinator()#创建一个队列
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(100002):
    # Retrieve a single instance:
    columns_value_batch = sess.run(columns_value_batch)
    print('第%d批数据'%(i))
    print(columns_value_batch)


  coord.request_stop()
  coord.join(threads)












