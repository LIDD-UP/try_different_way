# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: cnn_prediction_model.py
@time: 2018/11/16
"""
import tensorflow as tf
import numpy as np


class CNNModel(object):
    def __init__(self,n_input,n_output,stddev,
                 weights,biases,keep_ratio
                 ):
        # 模型的输入情况
        self.n_input = n_input,
        self.n_output = n_output,
        self.stddev = stddev,
        # 定义权重，字典的形式，因为有很多层卷积,权重有卷积和全连接的，池化是不需要的；
        '''
        遵循的字典形式：但是这里的卷积和池化都是可变的，所以需要一定的封装设计，这里先用简单的；
        weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=stddev)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=stddev))

        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
            'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
            'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
            'bd2': tf.Variable(tf.random_normal([n_output], stddev=stddev))
}
        '''
        self.weights = weights
        # 定义偏差，字典的形式，有很多层卷积
        self.biases = biases
        # 定义每一层卷积神经元的保留情况
        self.keep_ratio = keep_ratio
    # 这里可以考虑将卷积和池化，drop封装成一个函数
    def conv_pool_dr(self,weight,biases,keepratio):
        reshape_size = np.sqrt(self.n_input)
        _input = tf.reshape(self.n_input, [-1, reshape_size, reshape_size, 1])
        _conv1 = tf.nn.conv2d(_input, self.weights['wc1'], strides=[1, 1, 1, 1], padding="SAME")
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, self.biases['bc1']))
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv_pool_dr_result = tf.nn.dropout(_pool1, self.keep_ratio)
        return conv_pool_dr_result

    def full_connect_1(self):
        pass

    def full_connect_end(self):
        pass


    def cnn_net(self):
        # 首先reshape 输入的形状：
        # 正方形形状：
        reshape_size = np.sqrt(self.n_input)
        _input = tf.reshape(self.n_input,[-1,reshape_size,reshape_size,1])
        # 一层卷积
        _conv1 = tf.nn.conv2d(_input,self.weights['wc1'],strides=[1,1,1,1],padding="SAME")
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1,self.biases['bc1']))
        # 一层池化：
        _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        # dropout
        _pool_dr1 = tf.nn.dropout(_pool1,self.keep_ratio)

        # _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        # _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
        # _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
        # # 定义全连接层
        # _dense1 = tf.reshape(_pool_dr2, [-1, weights['wd1'].get_shape().as_list()[0]])
        # _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        # _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        #
        # out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
        #        'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
        #        'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
        #        }
        # return out






