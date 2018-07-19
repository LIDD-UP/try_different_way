#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_knowledge.py
@time: 2018/7/3
"""
'''
tf.estimator 的过程包括：
    1:载入数据方式很多，目前我还算能用的是：pandas，tensorflow自带的数据导入方式；还有一中numpy的导入的方式目前还不会；
    2：定义featureColumn：就是定义特征列
    3：定义regressor ：定义回归器：我这里的模型是wide-n-deep型的，也就是要定义：DNNlinearconbinedregressor(也就是DNN线性组合回归器）
    4：训练：利用回归器进行训练，estimator.train(input-fn,,,,)
    5:评估：也就是测试，以前在做的的时候一般把测试和预测搞在一起了
    6：预测：预测目前不知道怎么版；


关于tf.estimator 这个库：
1：首先他是对我们一般的神经网络以及，常见算法的分装，分为wide deep  和wide n deep 目前还没搞清楚他三者具体指代什么；
2：特征工程用于处理输入的特征以及标签；（也就是常说的数据预处理）
3：input-fn 函数用于返回x和y ，用于x,y 数据的训练，当然还包括了对测试，预测数据的输入；
        这些输入函数包含很多，我遇到过pandas_input_fn ,还有就是numpy_input_fn
4:最后的就是train
5：测试
6：预测
'''

'''
1:关于特征工程的实验：
它里面有一个numeric_column 的包，用于处理特征，也就是列明
具体函数参数如下（numeric_column(key,shape,default_value(tensorflow自带的也有默认值处理，这样可以解决NaN
    loss during ...的问题，pandas里面采用的利用DataFrame.dropna(axis=0）的处理方式
    dtype，normalizer_fn(对数据的转换，不限于，数据的归一化，还有比如取对数，指数，等方法对数据的处理）
    
'''
import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_numeric():

    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)

    price_transformed_tensor = price_column._get_dense_tensor(builder)

    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))
    #Returns a dense `Tensor` as input layer based on given `feature_columns`.
    price_transformed_tensor = feature_column.input_layer(price, [price_column])

    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor]))


'''       
从上面的结果可以看出，transform_fn 将所有的数值+2来处理了。
使用_LazyBuilder和inpu_layer来分别进行了测试.效果是一样的.
'''

'''
bucketized_column
bucketized_column(
    source_column,
    boundaries
)
分桶操作：如：boudaries是[0,5,10,20,120]会将数据分为：
(-inf,0)  (0,5),(5,10),(10,20),(20,120) （120，+inf)这6个桶；
分桶操作会将数据转换成one_hot形式的编码：
    具体使用如下：
    price = {'price':[[5.], [15.], [25.], [35.]]}
    price_colunm = tf.estimator.feature_num.numeric_colunm('price')
    buket_price = feature_num.bucketized_column(price_column,[0,10,20,30,40]
    price_buckt_tensor = feature_colum.input_layer(price,[buket_price]
    
'''

'''
categorical_column_with_vocabulary_list
categorical_column_with_vocabulary_list(
    key,
    vocabulary_list,
    dtype=None,
    default_value=-1,
    num_oov_buckets=0
)

key: feature名字
vocabulary_list: 对于category来说，进行转换的list.也就是category列表.
dtype: 仅仅string和int被支持，其他的类型是无法进行这个操作的.
default_value: 当不在vocabulary_list中的默认值，这时候num_oov_buckets必须是0.
num_oov_buckets: 用来处理那些不在vocabulary_list中的值，如果是0，那么使用default_value进行填充;如果大于0
，则会在[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets]这个区间上重新计算当前特征的值.
'''

'''
feature_num = 可以通过numeric_column和categorical_column_with_vocabulari_list 来获取，只不过，前者得到的是一个密集的tensor
后者得到的是一个稀疏的tensor
column._get_dense_tensor
column._get_dense_tensors

但是也可以转换成密集的形式：也是通过one_hot的形式，知识multi-hot
具体步骤如下：
feature_column.indicator_column(color_column)
color_dense_tensor = festure_column.input_layer(color_data,[color_column])

这里说明一下：input_layer 的参数
def input_layer(features,
                feature_columns,
                weight_collections=None,
                trainable=True,
                cols_to_vars=None):
                features也就是我们的数据是整体的数据，feature——columns数据列的名称；
                
                对于categorical_column_with_vocabulary_list来说返回的是sparser_tensor，注意 id_tensor 这个是有效的，另外一个是None. 对于线性模型来说是可以直接使用sparser_tensor的。然而，对于深度模型来说，需要将sparser转换成dense，所以也就有了indicator_column 这个函数的出现。indicator_column的作用就是将category产生的sparser tensor转换成dense tensor.

注意: 
* input_layer: 只接受dense tensor 
* tables_initializer: 在sparser的时候使用的，
如果不进行初始化会出现 Table not initialized. 
[Node: hash_table_Lookup = LookupTableFindV2 这样的异常

'''





'''
categorical_column_with_hash_bucket
categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=tf.string
)

当category的数量很多，也就无法使用指定category的方法来处理了
，那么，可以使用这种哈希分桶的方式来进行处理。比如，切词之后的句子，每一个词可以使用这种方式来处理. 使用 categorical_column_with_vocabulary_file 也是一种不错的选择，比如将词频高的拿出来。毕竟对于hash_bucket来说，
对于bucket_size的选取是个问题。
'''

'''
categorical_column_with_identity
categorical_column_with_identity(
    key,
    num_buckets,
    default_value=None
)

这是对连续的数字类的处理函数。比如 id 一共有10000个，
那么可以使用这种方式。但是如果多数没有被使用，那么还不如使用
 categorical_column_with_hash_bucket 进行重新处理。
'''

'''
embedding_column

embedding_column(
    categorical_column,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
embedding_column是嵌入列的意思；


'''

'''
weighted_categorical_column
weighted_categorical_column(
    categorical_column,
    weight_feature_key,
    dtype=tf.float32
)
'''

'''
linear_model
linear_model(
    features,
    feature_columns,
    units=1,
    sparse_combiner='sum',
    weight_collections=None,
    trainable=True
)

对所有特征进行线性加权操作.也就是加biases
'''
'''
crossed_column
组合特征，这仅仅适用于sparser特征.
产生的依然是sparsor特征.
'''


'''
对于我们的预测模型，我们应该用categorical进行分类编码（one-hot)
这种方式有bucketized_colunm 或者是categorical_colum_with_cocabulary_list
catetorical_column_with_hach_bucket等，
然后使用coross_column或者是embedding_column来组合这些经过处理的特征；
目前不知道这两个方式的区别；

存在的疑问：
权重的问题，他这里需要指定，而不是随机初始化；weighted_categorical_column
加权操作采用line_model 方法加biases

目前还是不知道他的权重参数是如何指定的，或者是如何变化的，按照他的原理是通过
weight_categorical_column来指定，如果不指定默认让权重都一样让人很迷惑；

还有就是loss值按步骤打印怎么做到；

精确度怎么打印，他基本都做了封装，接口是什么；
'''












