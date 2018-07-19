# -*- coding:utf-8 _*-
"""
@author:Administrator
@file: official_tutorial_based_on_tradetype.py
@time: 2018/7/18
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
dirname = os.path.dirname(os.getcwd())
filename = '\\use_estimator_new\\house_info_2018_2.csv'

# 读文件 # 注意此处的header位0，不知道为什么，莫名其妙的多了一行行的下标
data = pd.read_csv(dirname + filename, header=0)
print(data)
data = data.dropna()

X = data.ix[:,:-1]
y = data.ix[:,-1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# 处理日期数据
def date_processing(_date_data):
    list_date = list(_date_data['listingDate'])
    list_break_together = []
    for data in list_date:
        list_break = data.split('/')
        list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together, columns=['year', 'month', 'day'], dtype='float32')
    return date_data_after_processing


X_train_date_data = date_processing(X_train)
test_date_data = date_processing(X_test)

print(X_train_date_data)
print(test_date_data)

longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')  # 可以考虑使用分桶策略
month = tf.feature_column.numeric_column('month')
day = tf.feature_column.numeric_column('day')
# tradeTypeId = tf.feature_column.numeric_column('tradeTypeId')
# buildingTypeId = tf.feature_column.numeric_column('buildingTypeId')


price_bucket = tf.feature_column.bucketized_column(price,
                                                   [100000, 200000, 300000, 400000, 500000, 600000, 700000,
                                                    800000, 900000, 1000000])
province = tf.feature_column.categorical_column_with_hash_bucket('province', hash_bucket_size=10)
city = tf.feature_column.categorical_column_with_hash_bucket('city', hash_bucket_size=10)
address = tf.feature_column.categorical_column_with_hash_bucket('address', hash_bucket_size=10)
buildingTypeId = tf.feature_column.categorical_column_with_hash_bucket('buildingTypeId', hash_bucket_size=2)

# 经纬度：
longitude_bucket = tf.feature_column.bucketized_column(longitude, [-100, -60, -30, -10])
latitude_bucket = tf.feature_column.bucketized_column(latitude, [40, 43, 46, 49, 52, 55])

# 对年月日进行分桶操作：其中对年进行categorical_column 操作，对月份和号数进行buckized_column 操作
# year_categorical = tf.feature_column.categorical_column_with_vocabulary_list('year', ['2017', '2018'])
month_bucket = tf.feature_column.bucketized_column(month, [4, 7, 10])
day_bucket = tf.feature_column.bucketized_column(day, [11, 21])

# 定义基本特征和组合特征
base_columns = [
    price_bucket, province, city, address, buildingTypeId,
    month_bucket, day_bucket, longitude_bucket, latitude_bucket
]

crossed_columns = [tf.feature_column.crossed_column(
    ['province', 'city', 'address'], hash_bucket_size=10
),

    tf.feature_column.crossed_column(
        [longitude_bucket, latitude_bucket], 10
    )
]

deep_columns = [
    latitude,
    longitude,
    month, 
    day,
    price,

    # embedding将高纬的稀疏tensor转化成低维的tensor
    tf.feature_column.embedding_column(province, 8),
    tf.feature_column.embedding_column(city, 8),
    tf.feature_column.embedding_column(address, 8),
    tf.feature_column.embedding_column(buildingTypeId, 8),
    tf.feature_column.embedding_column(price_bucket, 8),

    # tf.feature_column.indicator_column(tradeTypeId),
    # tf.feature_column.indicator_column(year_categorical),

    # tf.feature_column.indicator_column(province),
    # tf.feature_column.indicator_column(city),
    # tf.feature_column.indicator_column(address),
    # tf.feature_column.indicator_column(buildingTypeId)
]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./tmp_official_tradetype2/predict_model',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_columns,
    # dnn_hidden_units= [2048,1024, 512, 256,128,64,32,16],
    # dnn_hidden_units= [32,64],
    dnn_hidden_units=[16, 32, 64, 128, 256, 512, 1024, 2048],
    linear_optimizer=tf.train.AdadeltaOptimizer(),
    dnn_optimizer= tf.train.AdamOptimizer()
    # dnn_optimizer= tf.train.ProximalAdagradOptimizer(
    # learning_rate=0.001
    # )
    #ev: {'average_loss': 1261.3373, 'loss': 157982.5, 'global_step': 10000}：30左右，这种模型处于瓶颈中，
    #需要改变优化函数，或者对特征或者数据进行进一步的处理
)

# 定义训练输入，测试输入，解决不同模型的输入对应问题

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(X_train['province']),
        'city': np.array(X_train['city']),
        'address': np.array(X_train['address']),
        'longitude': np.array(X_train['longitude']),
        'latitude': np.array(X_train['latitude']),
        'price': np.array(X_train['price']),
        'buildingTypeId': np.array(X_train['buildingTypeId']).astype('str'),
        'tradeTypeId': np.array(X_train['tradeTypeId']).astype('str'),
        # 'year': np.array(X_train_date_data['year']).astype('str'),
        'month': np.array(X_train_date_data['month']),
        'day': np.array(X_train_date_data['day'])
    },
    y=np.array(y_train),
    num_epochs=None,
    shuffle=False
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(X_test['province']),
        'city': np.array(X_test['city']),
        'address': np.array(X_test['address']),
        'longitude': np.array(X_test['longitude']),
        'latitude': np.array(X_test['latitude']),
        'price': np.array(X_test['price']),
        'buildingTypeId': np.array(X_test['buildingTypeId']).astype('str'),
        'tradeTypeId': np.array(X_test['tradeTypeId']).astype('str'),
        # 'year': np.array(test_date_data['year']).astype('str'),
        'month': np.array(test_date_data['month']),
        'day': np.array(test_date_data['day'])
    },
    y=np.array(y_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)

# 训练
# for j in range(30):
#     estimator_model.train(input_fn=train_input_fn,steps=1000)
estimator_model.train(input_fn=train_input_fn, steps=1000)

# 测试
ev = estimator_model.evaluate(input_fn=test_input_fn, steps=100)
print('ev: {}'.format(ev))
# 预测
predict_iter = estimator_model.predict(input_fn=test_input_fn)
# 循环次数根据测试数据数决定

list_value = []
for i in range(len(y_test)):
    # 目前不知道这个dict_values([array([51.575745], dtype=float32)]) 数据怎么把值弄出来，就没有算精确度了
    x =float(list(predict_iter.__next__().values())[0])
    print(i, x)
    list_value.append(x)

print(list_value)
print(mean_absolute_error(y_test,list_value))
