#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_read_data2.py
@time: 2018/7/23
"""

import tensorflow as tf
from sklearn import preprocessing
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import numpy as np
np.set_printoptions(suppress=True)

def read_data(file_queue):
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [0.], [0.], [0.], [0.], [0.], [''],[0.],[0.]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address,longitude,latitude, price, buildingTypeId,  tradeTypeId,  listingDate,  daysOnMarket,bedrooms = tf.decode_csv(value, record_defaults)
    return tf.stack([longitude,latitude, price, buildingTypeId,]),daysOnMarket



def create_pipeline(filename,batch_size,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example1, dayOnMarket1 = read_data(file_queue) # example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    min_after_dequeue = 1000  # 出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    capacity = min_after_dequeue+batch_size#队列的长度
    # example_batch,label_batch= tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)#把队列的数据打乱了读取
    example_batch,daysOnMarket_batch= tf.train.batch([example1,dayOnMarket1],batch_size=batch_size,capacity=capacity) # 顺序读取
    return example_batch, daysOnMarket_batch


# 定义网络
n_input = 4
n_output = 1
stddev = 0.1
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
    'wd1': tf.Variable(tf.random_normal([1 * 1 * 128, 1024], stddev=stddev)),
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
    _input_r = tf.reshape(_input, [-1, 2, 2, 1])
    # 卷积一层
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 池化第一层
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 随机丢弃一些点
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # 卷积二层，这里不池化
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    # _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_conv2, _keepratio)
    # 定义全连接层
    _dense1 = tf.reshape(_pool_dr2, [-1, weights['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2,  'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out,
           # 'pool2': _pool2,

           }
    return out


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

pred = conv_basic(x, weights, biases, keepratio)['out']

cost = tf.reduce_mean(tf.square(y-pred))
optimize = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimize.minimize(cost)



example_batch, daysOnMarket_batch = create_pipeline('month_4_1.csv',50)
example_batch_test,daysOnMarket_batch_test = create_pipeline('test_data_6_1.csv',30)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Start populating the filename queue.

    coord = tf.train.Coordinator()#创建一个队列
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10000):
        # Retrieve a single instance:
        example, label = sess.run([example_batch, daysOnMarket_batch])
        print('第%d批数据'%(i))
        print(example, label)
        example_1 = example.astype('float')
        print(example_1.dtype)
        label_1 = label.reshape(-1,1).astype('float')
        print(label_1.dtype)
        sess.run(train, feed_dict={x:example, y:label_1,keepratio:0.7})

    example_test_x ,label_test_y = sess.run([example_batch_test,daysOnMarket_batch_test])
    label_test_y_1 = label_test_y.reshape(-1,1)
    prediction_value =  sess.run(pred, feed_dict={x:example_test_x,y:label_test_y_1,keepratio:0.7})
    print(prediction_value)
    print(label_test_y_1)
    error = mean_absolute_error(label_test_y_1,prediction_value)
    print(error)

    coord.request_stop()
    coord.join(threads)

