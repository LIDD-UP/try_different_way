# -*- coding:utf-8 _*-
"""
@author:Administrator
@file: official_tutorial_5.py
@time: 2018/7/19
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error



'''
精度每0.001 为100米
维度每隔0，001 为111米；
0.005
0.001*(500/111)
对于精度大于0的删除：
对于维度小于30的删除；
'''
# 日志
tf.logging.set_verbosity(tf.logging.INFO)

dirname = os.path.dirname(os.getcwd())
train_filename = '\\try_different_way\\in_a_month_1.csv'
test_filename = '\\try_different_way\\test_data2_1.csv'

# 加载训练数据
data = pd.read_csv(dirname + train_filename, header=0)
# 去除经度大于0的：
data = data[data.longitude<0]
# 去除维度小于30的
data = data[data.latitude>30]
data = data.dropna(axis=0)
example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
                'listingDate','bedrooms']]
print(type(example))
label = data[['daysOnMarket']]

# 加载测试数据
data_test = pd.read_csv(dirname + test_filename, header=0)
data_test = data_test.dropna(axis=0)
example_test = data_test[
    ['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
      'listingDate','bedrooms']]
label_test = data_test[['daysOnMarket']]
print(example, label_test)

# 取出经纬度的最大值和最小值
longitude_min = int(example['longitude'].min())
longitude_max = int(example['longitude'].max())
latitude_min = int(example['latitude'].min())
latitude_max = int(example['latitude'].max())


# 处理日期把日期拆分成为年月日三列：
def date_processing(_date_data):
    list_date = list(_date_data['listingDate'])
    list_break_together = []
    for data in list_date:
        list_break = data.split('/')
        list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together, columns=['year', 'month', 'day'], dtype='float32')
    return date_data_after_processing



example_date_data = date_processing(example)
test_date_data = date_processing(example_test)



# 定义连续型连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')  # 可以考虑使用分桶策略
month = tf.feature_column.numeric_column('month')
day = tf.feature_column.numeric_column('day')

# 交易类型
tradeTypeId = tf.feature_column.categorical_column_with_vocabulary_list('tradeTypeId', ['1', '2'])
buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId', ['1', '2'])

# 由于受房屋类型以及交易类型的影响，价格的波动过于强烈，这里采用分桶的方式
# 分桶，对于价格再10000一下的基本时出租类型的数据；而对于10000，100000，200000，300000，400000，
# 500000，600000，700000，800000，900000，1000000
price_bucket = tf.feature_column.bucketized_column(price,
                                                   [500000, 1000000, 1500000,
                                                    2000000, 4000000])
province = tf.feature_column.categorical_column_with_hash_bucket('province', hash_bucket_size=100)
city = tf.feature_column.categorical_column_with_hash_bucket('city', hash_bucket_size=100)
address = tf.feature_column.categorical_column_with_hash_bucket('address', hash_bucket_size=100)


def generate_longtitude_and_latitude_list(min,max,distance):
    list_len = (max -min)/distance
    list_boundaries =[]
    middle = min
    for i in range(int(list_len)):
        middle += distance
        list_boundaries.append(middle)
    return list_boundaries


longitude_boudaries = generate_longtitude_and_latitude_list(longitude_min, longitude_max, 0.005)
latitude_boudaries = generate_longtitude_and_latitude_list(latitude_min, latitude_max, 0.005)

# longtitude_list = np.arange(-150,0,0.005)
# latitude_list = np.arange(40,60,0.005)

# 经纬度：
longitude_bucket = tf.feature_column.bucketized_column(longitude, sorted(longitude_boudaries))
latitude_bucket = tf.feature_column.bucketized_column(latitude, sorted(latitude_boudaries))

# 对年月日进行分桶操作：其中对年进行categorical_column 操作，对月份和号数进行buckized_column 操作
year_categorical = tf.feature_column.categorical_column_with_vocabulary_list('year', ['2017', '2018'])
month_bucket = tf.feature_column.bucketized_column(month, [4, 7, 10])
day_bucket = tf.feature_column.bucketized_column(day, [11, 21])

# 定义基本特征和组合特征
base_columns = [
    price_bucket,  tradeTypeId, province, city, address, buildingTypeId, year_categorical,
    month_bucket, day_bucket, longitude_bucket, latitude_bucket
]

crossed_columns = [tf.feature_column.crossed_column(
    ['province', 'city', 'address'], hash_bucket_size=1000
),

    tf.feature_column.crossed_column(
        [longitude_bucket, latitude_bucket], 1000
    )
]

deep_columns = [
    # price,
    latitude,
    longitude,


    # embedding将高纬的稀疏tensor转化成低维的tensor
    tf.feature_column.embedding_column(province, 8),
    tf.feature_column.embedding_column(city, 8),
    tf.feature_column.embedding_column(address, 8),
    tf.feature_column.embedding_column(price_bucket, 8),

    tf.feature_column.indicator_column(tradeTypeId),
    tf.feature_column.indicator_column(buildingTypeId),
    tf.feature_column.indicator_column(year_categorical),
    # tf.feature_column.indicator_column(day_bucket),
    # tf.feature_column.indicator_column(month_bucket),

    # tf.feature_column.indicator_column(province),
    # tf.feature_column.indicator_column(city),
    # tf.feature_column.indicator_column(address),
]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./tmp_official1/predict_model',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_columns,
    # dnn_hidden_units= [2048,1024, 512, 256,128,64,32,16],
    dnn_hidden_units= [32,64],
    # dnn_hidden_units=[16, 32, 64, 128, 256, 512, 1024, 2048],
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
        'province': np.array(example['province']),
        'city': np.array(example['city']),
        'address': np.array(example['address']),
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeId': np.array(example['buildingTypeId']).astype('str'),
        'tradeTypeId': np.array(example['tradeTypeId']).astype('str'),
        'year': np.array(example_date_data['year']).astype('str'),
        'month': np.array(example_date_data['month']),
        'day': np.array(example_date_data['day'])
    },
    y=np.array(label),
    num_epochs=None,
    shuffle=False
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(example_test['province']),
        'city': np.array(example_test['city']),
        'address': np.array(example_test['address']),
        'longitude': np.array(example_test['longitude']),
        'latitude': np.array(example_test['latitude']),
        'price': np.array(example_test['price']),
        'buildingTypeId': np.array(example_test['buildingTypeId']).astype('str'),
        'tradeTypeId': np.array(example_test['tradeTypeId']).astype('str'),
        'year': np.array(test_date_data['year']).astype('str'),
        'month': np.array(test_date_data['month']),
        'day': np.array(test_date_data['day'])
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)

# 训练
# for j in range(30):
#     estimator_model.train(input_fn=train_input_fn,steps=1000)ss
estimator_model.train(input_fn=train_input_fn, steps=3000)

# 测试
ev = estimator_model.evaluate(input_fn=test_input_fn, steps=100)
print('ev: {}'.format(ev))
# 预测
predict_iter = estimator_model.predict(input_fn=test_input_fn)
# 循环次数根据测试数据数决定

list_value = []
for i in range(len(label_test)):
    # 目前不知道这个dict_values([array([51.575745], dtype=float32)]) 数据怎么把值弄出来，就没有算精确度了
    x =float(list(predict_iter.__next__().values())[0])
    print(i, x)
    list_value.append(x)

print(list_value)
print('prediction_mean',np.mean(list_value))
print('label_mean',np.mean(label_test))

print(mean_absolute_error(label_test,list_value))

