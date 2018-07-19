#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: official_tutorial_2.py
@time: 2018/7/6
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np


# 数据的读入：
tf.logging.set_verbosity(tf.logging.INFO) #答应出日志记录观察到一条：INFO:tensorflow:Restoring parameters from ./models/dnnlregressor\model.ckpt-400

dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator_new\\house_info.csv'
test_filename = '\\use_estimator_new\\test_house_info.csv'
# 加载训练数据




data = pd.read_csv(dirname+train_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'daysOnMarket'])

data = data.dropna(axis=0)
example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice']]
print(type(example))

label = data[['daysOnMarket']]




data_test = pd.read_csv(dirname+test_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'daysOnMarket'])

data_test = data_test.dropna(axis=0)
example_test = data_test[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice']]
label_test = data_test[['daysOnMarket']]

print(example, label_test)






# 连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')  # 可以考虑使用分桶策略
expectedDealPrice =tf.feature_column.numeric_column('expectedDealPrice')  # 可以考虑使用分桶策略

tradeTypeName = tf.feature_column.categorical_column_with_vocabulary_list('tradeTypeName', ['Sale', 'Lease'])
# 由于受房屋类型以及交易类型的影响，价格的波动过于强烈，这里采用分桶的方式
# 分桶，对于价格再10000一下的基本时出租类型的数据；而对于10000，100000，200000，300000，400000，
#500000，600000，700000，800000，900000，1000000

price_bucket = tf.feature_column.bucketized_column(price, [10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
expectedDealPrice_bucket = tf.feature_column.bucketized_column(expectedDealPrice, [10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
province = tf.feature_column.categorical_column_with_hash_bucket('province',hash_bucket_size=100)
city = tf.feature_column.categorical_column_with_hash_bucket('city',hash_bucket_size=100)
address = tf.feature_column.categorical_column_with_hash_bucket('address',hash_bucket_size=100)
buildingTypeName = tf.feature_column.categorical_column_with_hash_bucket('buildingTypeName',hash_bucket_size=100)

# 定义基本特征和组合特征
base_columns = [
 price_bucket,expectedDealPrice_bucket, tradeTypeName, province, city, address,buildingTypeName
]


crossed_columns = [tf.feature_column.crossed_column(
    ['province', 'city', 'address'], hash_bucket_size=1000
)
   # tf.feature_column.crossed_column(
   #     [price, expectedDealPrice],1000
   # ), #由于crossed_columns只适用于sparse tensor 可以考虑进行bucktized_column 转化成one_hot形式的编码，也就是分区间（是一个量化的过程）
   #  tf.feature_column.crossed_column(
   #      [longitude, latitude], 1000
   #  ) ，这里也可以把经纬度也进行分区间，也就是某个城市，经纬度的分布问题，用bucktized_column也可以考虑一下；
]

deep_columns = [
    price,
    latitude,
    longitude,
    expectedDealPrice,
    # embedding将高纬的稀疏tensor转化成低维的tensor
    tf.feature_column.embedding_column(province,8),
    tf.feature_column.embedding_column(city,8),
    tf.feature_column.embedding_column(address,8),
    tf.feature_column.embedding_column(buildingTypeName,8),

    tf.feature_column.indicator_column(tradeTypeName)
    # tf.feature_column.indicator_column(province),
    # tf.feature_column.indicator_column(city),
    # tf.feature_column.indicator_column(address),
    # tf.feature_column.indicator_column(buildingTypeName)
]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./tmp/predict_model',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns = deep_columns,
    # dnn_hidden_units= [1024, 512, 256,128,64,32,16],
    dnn_hidden_units=[16,32,64,128,256,512,1024,2048]
    # linear_optimizer=tf.train.AdadeltaOptimizer(), # 对于稀疏的数据用自适应优化器更好；
    # dnn_optimizer= tf.train.AdamOptimizer()
)

# 定义训练输入，测试输入，解决不同模型的输入对应问题
# 'province', 'city', 'address', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeName', 'expectedDealPrice'

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={
        'province': np.array(example['province']),
        'city': np.array(example['city']),
        'address': np.array(example['address']),
        'longitude': np.array(example['longitude']),
        'latitude': np.array(example['latitude']),
        'price': np.array(example['price']),
        'buildingTypeName': np.array(example['buildingTypeName']),
        'tradeTypeName': np.array(example['tradeTypeName']),
        'expectedDealPrice': np.array(example['expectedDealPrice'])
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
        'buildingTypeName': np.array(example_test['buildingTypeName']),
        'tradeTypeName': np.array(example_test['tradeTypeName']),
        'expectedDealPrice': np.array(example_test['expectedDealPrice'])
       },
    y=np.array(label_test),
    num_epochs=1, # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)











# 训练
for j in range(60):
    estimator_model.train(input_fn=train_input_fn,steps=1000)
# 测试
ev = estimator_model.evaluate(input_fn=test_input_fn, steps=1)
print('ev: {}'.format(ev))
# 预测
predict_iter = estimator_model.predict(input_fn=test_input_fn)

# predict_iter_list = [x for x in predict_iter]

for i in range(27):
    # ratio = np.mean(sum())
    # print(i,'真实值:',label_test['daysOnMarket'][i])
    print(i,predict_iter.__next__().values())
    # print(help(predict_iter.__next__().values()))


'''
0 dict_values([array([23.70579], dtype=float32)])
1 dict_values([array([40.906166], dtype=float32)])
2 dict_values([array([5.573082], dtype=float32)])
3 dict_values([array([8.788507], dtype=float32)])
4 dict_values([array([6.1326733], dtype=float32)])
5 dict_values([array([58.66938], dtype=float32)])
6 dict_values([array([22.00718], dtype=float32)])
7 dict_values([array([24.900478], dtype=float32)])
8 dict_values([array([31.830687], dtype=float32)])
9 dict_values([array([13.323677], dtype=float32)])
10 dict_values([array([33.474503], dtype=float32)])
11 dict_values([array([9.572611], dtype=float32)])
12 dict_values([array([5.7744646], dtype=float32)])
13 dict_values([array([29.098503], dtype=float32)])
结果偏小了，
'''