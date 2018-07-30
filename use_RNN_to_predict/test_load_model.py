#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_load_model.py
@time: 2018/7/30
"""
import tensorflow as tf
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/my-model-1.meta')
    new_saver.restore(sess, './model/my-model-1')
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    y = tf.get_collection('pred_network')[0]

    graph = tf.get_default_graph()
    # print(graph)

    # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
    input_x = graph.get_operation_by_name('inputs/xs').outputs[0]

    seq, res = get_batch_boston()
    # 使用y进行预测
    result = sess.run(y, feed_dict={input_x: seq})
    print(result)