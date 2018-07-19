#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: use_estimator_compensation.py
@time: 2018/7/2
"""
'''
梳理tf,estimator.DNNlinearnregression

'''



# tensorflow又把contrib.learn包移动到estimator下了
# tf.estimator.DNNLinearCombinedRegressor 广度深度回归
# tf.estimator.DNNRegressor 神经网络回归器

import tensorflow as tf
import pandas as pd
import os
import numpy as np
import logging

tf.logging.set_verbosity(tf.logging.INFO) #答应出日志记录观察到一条：INFO:tensorflow:Restoring parameters from ./models/dnnlregressor\model.ckpt-400

dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator_new\\house_info.csv'
test_filename = '\\use_estimator_new\\test_house_info.csv'
# 加载训练数据
data = pd.read_csv(dirname+train_filename, header=1, usecols=[4, 5, 6, 11, 14], names=['longitude', 'latitude','price','expectedDealPrice','daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
# print(data[['longitude']]) #他是一个二位的数组，需要有两个【】这个；
data = data.dropna(axis=0)
example = data[['longitude', 'latitude', 'price','expectedDealPrice']]
label = data[['daysOnMarket']]



# 加载测试数据
data_test = pd.read_csv(dirname+test_filename,header=1,usecols=[4,5,6,11,14] ,names=['longitude','latitude','price','expectedDealPrice','daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
data_test = data_test.dropna(axis=0)
example_test = data_test[['longitude', 'latitude', 'price','expectedDealPrice']]
label_test = data_test[['daysOnMarket']]

print(example,label_test)

# feature_cols = [tf.contrib.layers.real_valued_column(k) for k in example] #定义特征列； 这种方法不太确定，用另外一种方法
# print(feature_cols)


FEATURE_COLUMNS =[
 'longitude', 'latitude', 'price', 'expectedDealPrice'
]


#定义模型；
estimator = tf.estimator.DNNLinearCombinedRegressor(
                                linear_feature_columns=[tf.feature_column.numeric_column('longitude'),
                                tf.feature_column.numeric_column('latitude'),
                                tf.feature_column.numeric_column('price'),
                                tf.feature_column.numeric_column('expectedDealPrice'),
                                                        ],
                                dnn_hidden_units=[4,16,32],
                                model_dir='./models_test/dnnlregressor',
                                dnn_feature_columns=[tf.feature_column.numeric_column('price')],
                                linear_optimizer=tf.train.ProximalAdagradOptimizer(
                                learning_rate=0.1,
                                l1_regularization_strength=0.001
                                            )
                                            )

def input_fn(df, label):
    feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
    label = tf.constant(label.values)
    return feature_cols, label


def train_input_fn():
    '''训练阶段使用的 input_fn'''
    return input_fn(example, label)


def test_input_fn():
    '''测试阶段使用的 input_fn'''
    return input_fn(example_test, label_test)








# 训练
estimator.train(input_fn=train_input_fn,steps=100)
# 测试
ev = estimator.evaluate(input_fn=test_input_fn, steps=1)

print('ev: {}'.format(ev))
# 预测
predict_iter = estimator.predict(input_fn=test_input_fn)

for predict in predict_iter:
    print(predict['predictions'])






