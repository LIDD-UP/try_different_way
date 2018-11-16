#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: CNN_prediction_day.py
@time: 2018/7/23
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data



def read_data(file_queue):
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [''], [0.], [0.], [0.], [0.], [''],[0], [''], [0.], [''], [''], [0]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address, postCode, longitude,latitude, price, buildingTypeId, buildingTypeName, tradeTypeId, tradeTypeName, expectedDealPrice, listingDate, delislingDate, daysOnMarket = tf.decode_csv(value, record_defaults)
    return tf.stack([price,expectedDealPrice]),daysOnMarket



#批量获取
def create_pipeline(filename,batch_size,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
    example,dayOnMarket = read_data(file_queue)#example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    min_after_dequeue = 1000#出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    capacity = min_after_dequeue+batch_size#队列的长度
    #example_batch,label_batch= tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)#把队列的数据打乱了读取
    example_batch,daysOnMarket_batch= tf.train.batch([example,dayOnMarket],batch_size=batch_size,capacity=capacity)#顺序读取

    return example_batch,daysOnMarket_batch






mnist = input_data.read_data_sets('data/', one_hot=True)  # 此处的one_hot =True必须加上，不然可能会导致，形状有问题，在做矩阵乘法时出错；
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

n_input = 784
n_output = 10
stddev = 0.1
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


# 定义卷积层模型：与一般的神经网络模型多了卷积和池化

def conv_basic(_input, _w, _b, _keepratio):
    # 改变输入的形状为28*28*1*1
    _input_r = tf.reshape(_input, [-1, 28, 28, 1])
    # 卷积一层
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 池化第一层
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 随机丢弃一些点
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # 卷积二层，池化二层
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # 定义全连接层
    _dense1 = tf.reshape(_pool_dr2, [-1, weights['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

pred = conv_basic(x, weights, biases, keepratio)['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimize.minimize(cost)
accr1 = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accr = tf.reduce_mean(tf.cast(accr1, 'float'))

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    train_epochs = 15
    batch_size = 16
    avg_cost = 0
    total_batch = 10
    for epoch in range(train_epochs):
        for i in range(total_batch):
            example_batch, label_batch = mnist.train.next_batch(batch_size)
            feed_train = {x: example_batch, y: label_batch, keepratio: 0.7}
            sess.run(train, feed_dict=feed_train)
            avg_cost += sess.run(cost, feed_dict=feed_train)
        avg_cost = avg_cost / total_batch

        if (epoch + 1) % 2:
            feed_train = {x: example_batch, y: label_batch, keepratio: 1.}
            # feed_test = {x:test_img,y:test_label}
            print('train accr：%f' % sess.run(accr, feed_dict=feed_train))
            # print('test accr'%sess.run(accr,feed_dict=feed_test))

example_batch, daysOnMarket_batch = create_pipeline(train_filename, 10)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
print('.........................>>>>开始会话')
# 创建会话，采用上下文管理器的方式，无需手动关闭会话
with tf.Session() as sess:
    sess.run(init_op)
    # 创建一个队列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(100):
        #获取正真的样本和标签
        example, label = sess.run([example_batch, daysOnMarket_batch])
        print('第%d批数据'%(step))
        print(example, label)
        print('.......这一批数据的直接参数')
        reg = linear_model.LinearRegression()
        reg.fit(example, label)
        print("Coefficients of sklearn: W=%s, b=%f" % (reg.coef_, reg.intercept_))
        # 数据归一化处理
        scaler = preprocessing.StandardScaler().fit(example)
        print(scaler.mean_, scaler.scale_)
        x_data_standard = scaler.transform(example)

        sess.run(train, feed_dict={x_data: x_data_standard, y_data: label})
        # 每十步获取一次w和b
        if step % 10 == 0:
            print('当前w值和b值')
            print(sess.run(w, feed_dict={x_data: x_data_standard, y_data: label}),
                  sess.run(b, feed_dict={x_data: x_data_standard, y_data: label}))
    print('。。。。。。。》》》训练后得到w和b')
    theta = sess.run(w)
    intercept = sess.run(b).flatten()
    print('W:%s' % theta)
    print('b:%f' % intercept)
    coord.request_stop()
    coord.join(threads)

