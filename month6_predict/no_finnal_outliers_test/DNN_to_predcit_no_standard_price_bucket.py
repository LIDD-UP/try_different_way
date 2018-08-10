#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: DNN_to_predcit.py
@time: 2018/8/7
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 日志
tf.logging.set_verbosity(tf.logging.INFO)
train_data = pd.read_csv('./processing_data/no_standard_data/no_final_Outliers_no_Standard_train.csv')
print(train_data.shape)
print(train_data.head())

test_data = pd.read_csv('./processing_data/no_standard_data/no_final_Outliers_no_Standard_test_1.csv')
train_data = train_data.dropna()
print(train_data.shape)
test_data = test_data.dropna()

def data_processing(train_data):
    train_data['longitude'] = StandardScaler().fit_transform(np.array(train_data['longitude']).reshape(-1,1))
    train_data['latitude'] = StandardScaler().fit_transform(np.array(train_data['latitude']).reshape(-1,1))
    train_data['price'] = StandardScaler().fit_transform(np.array(train_data['price']).reshape(-1,1))
    return train_data

train_data = data_processing(train_data)
test_data = data_processing(test_data)


example = train_data[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]
example_test = test_data[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]

ss = StandardScaler()
label = ss.fit_transform(np.array(train_data['daysOnMarket']).reshape(-1,1))
label_test = test_data['daysOnMarket']



# 定义连续型连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')

# 交易类型和房间数
buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId', [1, 2])
bedrooms = tf.feature_column.categorical_column_with_vocabulary_list('bedrooms',[1,2,3,4,5])



deep_columns = [
    price,
    latitude,
    longitude,
    tf.feature_column.indicator_column(buildingTypeId),
    tf.feature_column.indicator_column(bedrooms),

]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNRegressor(
    model_dir='./DNN_no_standard_price_bucket',
    feature_columns=deep_columns,
    # hidden_units=[1024,512, 256, 128, 64, 32],
    hidden_units=[16,32,64,128,256, 512,256,128,64,32,16],

    # hidden_units=[32,64],
    # hidden_units=[64,32],
    # dropout=0.01,
    # optimizer=tf.train.AdamOptimizer(),
)

batch_size = 10

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeId': np.array(example['buildingTypeId']).astype(int),
        'bedrooms': np.array(example['bedrooms']).astype(int)
    },
    y=np.array(label),
    num_epochs=None,
    shuffle=True,
    batch_size=batch_size,
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'longitude': np.array(example_test['longitude']),
        'latitude': np.array(example_test['latitude']),
        'price': np.array(example_test['price']),
        'buildingTypeId': np.array(example_test['buildingTypeId']).astype(int),
        'bedrooms': np.array(example_test['bedrooms']).astype(int)
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False,
    batch_size=batch_size,
)

# 训练
steps_trains = int(len(example)/batch_size)
print(steps_trains)
steps_test = int(len(example_test)/batch_size)
#
# for i in range(10000):
#     estimator_model.train(input_fn=train_input_fn, steps=steps_trains)
estimator_model.train(input_fn=train_input_fn, steps=steps_trains)

# 测试
# ev = estimator_model.evaluate(input_fn=test_input_fn, steps=steps_test)
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
list_value = np.array(list_value)
list_value = ss.inverse_transform(list_value)

print('prediction_mean',np.mean(list_value))
print('label_mean',np.mean(label_test))

print(mean_absolute_error(label_test,list_value))

plt.plot(list_value,label='preds',c='red')
plt.plot(label_test,label='true',c='blue')
plt.show()
