#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: use_estimator.py
@time: 2018/7/2
"""

import tensorflow as tf
import os
import numpy as np
import math


def read_data(file_queue):
    '''
    the function is to get features and label (即样本特征和样本的标签）
    数据来源是csv的文件，采用tensorflow 自带的对csv文件的处理方式
    :param file_queue:
    :return: features,label
    '''
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [''], [0.], [0.], [0.], [0.], [''],[0], [''], [0.], [''], [''], [0]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address, postCode, longitude,latitude, price, buildingTypeId, buildingTypeName, tradeTypeId, tradeTypeName, expectedDealPrice, listingDate, delislingDate, daysOnMarket = tf.decode_csv(value, record_defaults)

    features = tf.stack([latitude,longitude,price,expectedDealPrice])
    return features, daysOnMarket


def create_pipeline(filename,batch_size,num_epochs=None):
    '''
    the function is to get every batch example and label
    此处使用的是tf.train.batch，即顺序获取，非随机获取，随机获取采用的方法是：tf.train.shuffle_batch
    :param filename:
    :param batch_size:
    :param num_epochs:
    :return:example_batch,label_batch
    '''
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
    # example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    example,dayOnMarket = read_data(file_queue)
    # 出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    min_after_dequeue = 1000
    #队列的长度
    capacity = min_after_dequeue+batch_size
    # 顺序获取每一批数据
    example_batch,daysOnMarket_batch = tf.train.batch([example,dayOnMarket],batch_size=batch_size,capacity=capacity)#顺序读取
    return example_batch,daysOnMarket_batch

train_example_batch ,train_label_batch= create_pipeline('house_info.csv',10)
test_example_batch ,test_example_batch = create_pipeline('test_house_info,csv',10)


'''
模型的权重参数如何定义；

'''
batch_size =10
hidden_units= [32, 64, 128]
cwd = os.getcwd()
dir_path = './save_path'
os.path.join(cwd ,dir_path)
weight_column1 = tf.feature_column.numeric_column('w1',shape=[4,32])
weight_column2 = tf.feature_column.numeric_column('w1',shape=[32,64])
weight_column3 = tf.feature_column.numeric_column('w1',shape=[64,128])

estimator = tf.estimator.DNNLinearCombinedRegressor(
    linear_feature_columns=[tf.feature_column.numeric_column('x', shape=[10,4])],
    linear_optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001
    ),
    dnn_hidden_units=hidden_units,
    model_dir = dir_path,
    dnn_activation_fn= tf.nn.relu

)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':np.array(train_example_batch)},
    y= np.array(train_label_batch),
    batch_size=batch_size,
    num_epochs =1,
    shuffle=False
)

predict_training_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':np.array(test_example_batch)},
    y= np.array(test_example_batch),
    batch_size=batch_size,
    num_epochs =1,
    shuffle=False
)

for period in range(100):
    estimator.train(
        input_fn = predict_training_input_fn,
        steps = 20

    )
    train_predict = estimator.predict(input_fn=predict_training_input_fn)
    training_root_mean_squared_error = math.sqrt(
        tf.matmul()
    )




