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
train_filename = '\\month_4_1.csv'
test_filename = '\\test_data_6_1.csv'

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

def normalization_price(data):
    minmax = MinMaxScaler()
    data_price= np.array(data[['longitude','price','latitude','buildingTypeId']])
    data[['longitude', 'price', 'latitude', 'buildingTypeId']] = minmax.fit_transform(data_price)
    return data
example = normalization_price(example)


# 加载测试数据
data_test = pd.read_csv(dirname + test_filename, header=0)
data_test = data_test.dropna(axis=0)
example_test = data_test[
    ['province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
      'listingDate','bedrooms']]
label_test = data_test[['daysOnMarket']]
print(example, label_test)
example_test = normalization_price(example_test)


longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')
buildingTypeId = tf.feature_column.numeric_column('buildingTypeId')

# DNN特征
deep_columns = [
    longitude,
    latitude,
    price,
    buildingTypeId
]

# estimator
estimator_model = tf.estimator.DNNRegressor(
    model_dir='./third/predict_model',
    feature_columns=deep_columns,
    # hidden_units=[512, 256, 128, 64, 32],
    hidden_units=[512, 256, 128, 64, 32],
)

# 训练输入
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeId': np.array(example['buildingTypeId'])
    },
    y=np.array(label),
    num_epochs=None,
    shuffle=False
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'longitude': np.array(example_test['longitude']),
        'latitude': np.array(example_test['latitude']),
        'price': np.array(example_test['price']),
        'buildingTypeId': np.array(example_test['buildingTypeId'])
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)

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


