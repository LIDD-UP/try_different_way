#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: daysonmarkt_DNN.py
@time: 2018/7/20
"""
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




tf.logging.set_verbosity(tf.logging.INFO)

dirname = os.path.dirname(os.getcwd())
train_filename = '\\try_different_way\\month_4_1.csv'
test_filename = '\\try_different_way\\test_data_6_1.csv'

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

# 对价格做映射也就是归一化：
def normalization_price(data):
    minmax = MinMaxScaler()
    data_price= np.array(data['price']).reshape(-1,1)
    data['price'] = minmax.fit_transform(data_price)
    return data
example = normalization_price(example)
print(example['price'])

# 加载测试数据
data_test = pd.read_csv(dirname + test_filename, header=0)
data_test = data_test.dropna(axis=0)
example_test = data_test[
    ['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
      'listingDate','bedrooms']]
label_test = data_test[['daysOnMarket']]
print(example, label_test)


def normalization_price(data):
    minmax = MinMaxScaler()
    data_price= np.array(data['price']).reshape(-1,1)
    data['price'] = minmax.fit_transform(data_price)
    return data
example_test = normalization_price(example_test)
print(example_test['price'])

# 取出经纬度的最大值和最小值
longitude_min = int(example['longitude'].min())
longitude_max = int(example['longitude'].max())
latitude_min = int(example['latitude'].min())
latitude_max = int(example['latitude'].max())

# 生成经纬度500*500的格式
def generate_longtitude_and_latitude_list(min,max,distance):
    list_len = (max -min)/distance
    list_boundaries =[]
    middle = min
    for i in range(int(list_len)):
        middle += distance
        list_boundaries.append(middle)
    return list_boundaries

# 生成经纬度实例范围
longitude_boudaries = generate_longtitude_and_latitude_list(longitude_min, longitude_max, 0.005)
latitude_boudaries = generate_longtitude_and_latitude_list(latitude_min, latitude_max, 0.005*(500/111))


# 处理日期把日期拆分成为年月日三列：
def date_processing(_date_data):
    list_date = list(_date_data['listingDate'])
    list_break_together = []
    for data in list_date:
        list_break = data.split('/')
        list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together, columns=['year', 'month', 'day'], dtype='float32')
    return date_data_after_processing

# 生成年月日实例
example_date_data = date_processing(example)
test_date_data = date_processing(example_test)


longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')
# month = tf.feature_column.numeric_column('month')
day = tf.feature_column.numeric_column('day')

# 交易类型
# tradeTypeId = tf.feature_column.categorical_column_with_vocabulary_list('tradeTypeId', ['1', '2'])
buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId', [1, 2])

# 经纬度：
longitude_bucket = tf.feature_column.bucketized_column(longitude, sorted(longitude_boudaries))
latitude_bucket = tf.feature_column.bucketized_column(latitude, sorted(latitude_boudaries))

# 对年月日进行分桶操作：其中对年进行categorical_column 操作，对月份和号数进行buckized_column 操作
month_bucket = tf.feature_column.categorical_column_with_vocabulary_list('month', [1,2,3,4,5,6,7,8,9,10,11,12])
day_bucket = tf.feature_column.bucketized_column(day, [11, 21])


# 组合特征

longtitude_latitude = tf.feature_column.crossed_column(
        [longitude_bucket, latitude_bucket], 1000
    )


# DNN特征
deep_columns = [
    longitude_bucket,
    latitude_bucket,
    price,
    tf.feature_column.embedding_column(longtitude_latitude,8),
    tf.feature_column.indicator_column(buildingTypeId),
    tf.feature_column.indicator_column(month_bucket),
    day_bucket,
]


estimator_model = tf.estimator.DNNRegressor(
    model_dir='./tmp_dnn_map_price/predict_model',
    feature_columns=deep_columns,
    # hidden_units=[512, 256, 128, 64, 32],
    hidden_units=[512, 256, 128, 64, 32],

)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        # 'province': np.array(example['province']),
        # 'city': np.array(example['city']),
        # 'address': np.array(example['address']),
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeId': np.array(example['buildingTypeId']).astype('int'),
        # # 'tradeTypeId': np.array(example['tradeTypeId']).astype('str'),
        # # 'year': np.array(example_date_data['year']).astype('str'),
        'month': np.array(example_date_data['month']).astype('int'),
        'day': np.array(example_date_data['day'])
    },
    y=np.array(label),
    num_epochs=None,
    shuffle=False
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        # 'province': np.array(example_test['province']),
        # 'city': np.array(example_test['city']),
        # 'address': np.array(example_test['address']),
        'longitude': np.array(example_test['longitude']),
        'latitude': np.array(example_test['latitude']),
        'price': np.array(example_test['price']),
        'buildingTypeId': np.array(example_test['buildingTypeId']).astype('int'),
        # 'tradeTypeId': np.array(example_test['tradeTypeId']).astype('str'),
        # # 'year': np.array(test_date_data['year']).astype('str'),
        'month': np.array(test_date_data['month']).astype('int'),
        'day': np.array(test_date_data['day'])
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)

# 训练
# for j in range(30):
#     estimator_model.train(input_fn=train_input_fn,steps=1000)ss
estimator_model.train(input_fn=train_input_fn, steps=1000)

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


