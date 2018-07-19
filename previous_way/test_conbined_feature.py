#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_conbined_feature.py
@time: 2018/7/3
"""

import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder

# 对于特征1：
price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本

price_column = feature_column.numeric_column('price')
bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])

price_bucket_tensor = feature_column.input_layer(price, [bucket_price])

# 对于特征2：
price2 = {'price2': [[5.], [15.], [25.], [35.]]}  # 4行样本

price_column2 = feature_column.numeric_column('price2')
bucket_price2 = feature_column.bucketized_column(price_column2, [0, 10, 20, 30, 40])

price_bucket_tensor2 = feature_column.input_layer(price2, [bucket_price2])

embed_column = feature_column.embedding_column(price_column,price_column2)
pass



# result = tf.concat(price_bucket_tensor2,price_bucket_tensor)
print(price_bucket_tensor.shape)
print(type(price_bucket_tensor))
# result = tf.stack([price_bucket_tensor,price_bucket_tensor2],axis=1)
#one_hot编码之后的张量进行合并，
result = tf.concat([price_bucket_tensor,price_bucket_tensor2],1)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    #print(session.run([price_bucket_tensor,price_bucket_tensor2]))
    print(result.eval())


