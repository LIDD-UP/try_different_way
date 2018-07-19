#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_bucktized_column_type.py
@time: 2018/7/5
"""

import tensorflow as tf
from tensorflow import feature_column

def test_bucketized_column():

    data= {'price': [[5.], [15.], [25.], [35.]],'price2': [[5.], [15.], [25.], [35.]]}  # 4行样本

    price_column = feature_column.numeric_column('price')
    price_column2 = feature_column.numeric_column('price2')
    print(price_column)
    bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])
    bucket_price2 = feature_column.bucketized_column(price_column2, [0, 10, 20, 30, 40])
    print(bucket_price)

    price_bucket_tensor = feature_column.input_layer(data, [bucket_price,bucket_price2])

    print(type(price_bucket_tensor))

    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

test_bucketized_column()