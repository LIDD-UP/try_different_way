#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_read_data.py
@time: 2018/7/23
"""

import tensorflow as tf
import os
dirname = os.path.dirname(os.getcwd())
train_filename = '\\month_4_1.csv'
test_filename = '\\test_data_6_1.csv'
train_file = dirname+train_filename
test_file = dirname+test_filename

def read_data(file_queue):
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [0.], [0.], [0.], [0.], [0.], [''],[0.],  [0.]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address,longitude,latitude, price, buildingTypeId,  tradeTypeId,  listingDate,  daysOnMarket,bedrooms = tf.decode_csv(value, record_defaults)
    return tf.stack([price,latitude]),daysOnMarket



#批量获取
def create_pipeline(filename,batch_size,num_epochs=None):
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
    example,dayOnMarket = read_data(file_queue)#example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    min_after_dequeue = 1000#出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    capacity = min_after_dequeue+batch_size#队列的长度
    #example_batch,label_batch= tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)#把队列的数据打乱了读取
    example_batch,daysOnMarket_batch= tf.train.batch([example,dayOnMarket],batch_size=batch_size,capacity=capacity)#顺序读取

    return example_batch,daysOnMarket_batch



example_batch, daysOnMarket_batch = create_pipeline(train_filename, 10)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
print('.........................>>>>开始会话')
# 创建会话，采用上下文管理器的方式，无需手动关闭会话
with tf.Session() as sess:
    sess.run(init_op)
    # 创建一个队列
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    for step in range(10):
        #获取正真的样本和标签
        example, label = sess.run([example_batch, daysOnMarket_batch])
        print('第%d批数据'%(step))
        print(example, label)
        print('.......这一批数据的直接参数')

    # coord.request_stop()
    # coord.join(threads)

