# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: previous_dnn.py
@time: 2018/9/12
"""

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


# 日志
tf.logging.set_verbosity(tf.logging.INFO)




# 加载训练数据
data = pd.read_csv('./input/train.csv')
data = data.dropna(axis=0)
def process_bedrooms(data):
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    return data
data = process_bedrooms(data)

# 去除经度大于0的：


example = data[['province', 'city',
                # 'address',
                'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
                'listingDate','bedrooms']]
print(type(example))
label = data[['daysOnMarket']]

# 加载测试数据
data_test = pd.read_csv('./input/test.csv')

data_test = data_test.dropna(axis=0)
data_test = process_bedrooms(data_test)
example_test = data_test[
    ['province', 'city',
     # 'address',
     'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
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
        list_break = data.split('-')
        list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together, columns=['year', 'month', 'day'], dtype='float32')
    return date_data_after_processing



example_date_data = date_processing(example)
test_date_data = date_processing(example_test)



# 定义连续型连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')
month = tf.feature_column.numeric_column('month')
day = tf.feature_column.numeric_column('day')
bedrooms = tf.feature_column.numeric_column('bedrooms')

# 交易类型
buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId', [1, 2])

price_bucket = tf.feature_column.bucketized_column(price,
                                                   [500000, 1000000, 1500000,
                                                    2000000, 4000000])

province = tf.feature_column.categorical_column_with_hash_bucket('province', hash_bucket_size=100)
city = tf.feature_column.categorical_column_with_hash_bucket('city', hash_bucket_size=100)
# address = tf.feature_column.categorical_column_with_hash_bucket('address', hash_bucket_size=100)


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

# 经纬度：
longitude_bucket = tf.feature_column.bucketized_column(longitude, sorted(longitude_boudaries))
latitude_bucket = tf.feature_column.bucketized_column(latitude, sorted(latitude_boudaries))

#month and day
month_bucket = tf.feature_column.categorical_column_with_vocabulary_list('month', [1,2,3,4,5,6,7,8,9,10,11,12])
day_bucket = tf.feature_column.bucketized_column(day, [11, 21])

# 定义基本特征和组合特征

longitude_latitude = tf.feature_column.crossed_column(
        [longitude_bucket, latitude_bucket], 1000
    )

base_columns = [
    price_bucket,
    province, city,
    # address,
    buildingTypeId,
    month_bucket, day_bucket,
    latitude, longitude_bucket,
    longitude, latitude_bucket,
    longitude_latitude,
    bedrooms
]

deep_columns = [
    price,
    latitude,
    longitude,
    bedrooms,

    day_bucket,
    price_bucket,

    tf.feature_column.embedding_column(province, 8),
    tf.feature_column.embedding_column(city, 8),
    # tf.feature_column.embedding_column(address, 8),
    tf.feature_column.embedding_column(longitude_latitude, 8),

    tf.feature_column.indicator_column(buildingTypeId),
    tf.feature_column.indicator_column(month_bucket),
]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./previous/predict_model',
    linear_feature_columns=base_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[1024,512, 256, 128, 64, 32],
    linear_optimizer=tf.train.AdadeltaOptimizer(),
    # dnn_optimizer=tf.train.AdamOptimizer()
)



train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(example['province']),
        'city': np.array(example['city']),
        # 'address': np.array(example['address']),
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeId': np.array(example['buildingTypeId']).astype('int'),
        'month': np.array(example_date_data['month']).astype('int'),
        'day': np.array(example_date_data['day']),
        'bedrooms': np.array(example['bedrooms'])
    },
    y=np.array(label),
    num_epochs=None,
    shuffle=False,
    batch_size=10,
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(example_test['province']),
        'city': np.array(example_test['city']),
        # 'address': np.array(example_test['address']),
        'longitude': np.array(example_test['longitude']),
        'latitude': np.array(example_test['latitude']),
        'price': np.array(example_test['price']),
        'buildingTypeId': np.array(example_test['buildingTypeId']).astype('int'),
        'month': np.array(test_date_data['month']).astype('int'),
        'day': np.array(test_date_data['day']),
        'bedrooms': np.array(example_test['bedrooms'])
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)

# 训练
for j in range(500):
    # estimator_model.train(input_fn=train_input_fn,steps=1000)ss
    estimator_model.train(input_fn=train_input_fn, steps=len(example)/10+1)

# # 测试
# ev = estimator_model.evaluate(input_fn=test_input_fn, steps=100)
# print('ev: {}'.format(ev))
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

