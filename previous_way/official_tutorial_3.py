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
from sklearn.metrics import mean_absolute_error

'''
对于categorical_column_with_hash_bucket是我们指定他又多少列；
到目前为止，我们处理的示例都包含很少的类别。
例如，我们的 product_class 示例只有 3 个类别。但是通常，类别的数量非常大，
以至于无法为每个词汇或整数设置单独的类别，因为这会消耗太多内存。
对于此类情况，我们可以反问自己：“我愿意为我的输入设置多少类别？
虽然说这样会导致一些不同的东西映射到一起，但是机器学习可以通过其他的特征被他们错误分类的东西分开；

还可以将经纬度在进行优化，把他两个组合起来进一步优化模型；再进行组合之前可以将经纬度先进行分桶；
把年通过分类标识列来解决；

其实可以这样理解，buketized_column，和categorical_column_with_identity其实对数字的分桶，
而categorial_column_with_vocabulary_list 和hash_bucket是进行的类别分桶；




'''






# 数据的读入：
tf.logging.set_verbosity(tf.logging.INFO) #答应出日志记录观察到一条：INFO:tensorflow:Restoring parameters from ./models/dnnlregressor\model.ckpt-400

dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator_new\\house_info.csv'
test_filename = '\\use_estimator_new\\test_house_info.csv'

# 加载训练数据
data = pd.read_csv(dirname+train_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,12,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'listingDate','daysOnMarket'])

data = data.dropna(axis=0)
example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice','listingDate']]
print(type(example))
label = data[['daysOnMarket']]

# 加载测试数据
data_test = pd.read_csv(dirname+test_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,12,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice' ,'listingDate','daysOnMarket'])
data_test = data_test.dropna(axis=0)
example_test = data_test[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice','listingDate']]
label_test = data_test[['daysOnMarket']]
print(example, label_test)


# 处理日期把日期拆分成为年月日三列：
def date_processing(_date_data):
    list_date = list(_date_data['listingDate'])
    list_break_together = []
    for data in list_date:
            list_break = data.split('/')
            list_break_together.append(list_break)
    date_data_after_processing = pd.DataFrame(list_break_together,columns=['year','month','day'],dtype='float32')
    return date_data_after_processing


example_date_data = date_processing(example)
test_date_data = date_processing(example_test)

# print(example_date_data['year'])
# pass


# 定义连续型连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')  # 可以考虑使用分桶策略
expectedDealPrice =tf.feature_column.numeric_column('expectedDealPrice')  # 可以考虑使用分桶策略
# year = tf.feature_column.numeric_column('year')
month = tf.feature_column.numeric_column('month')
day = tf.feature_column.numeric_column('day')

# 交易类型
tradeTypeName = tf.feature_column.categorical_column_with_vocabulary_list('tradeTypeName', ['Sale', 'Lease'])

# 由于受房屋类型以及交易类型的影响，价格的波动过于强烈，这里采用分桶的方式
# 分桶，对于价格再10000一下的基本时出租类型的数据；而对于10000，100000，200000，300000，400000，
# 500000，600000，700000，800000，900000，1000000
price_bucket = tf.feature_column.bucketized_column(price, [10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
expectedDealPrice_bucket = tf.feature_column.bucketized_column(expectedDealPrice, [10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
province = tf.feature_column.categorical_column_with_hash_bucket('province',hash_bucket_size=100)
city = tf.feature_column.categorical_column_with_hash_bucket('city',hash_bucket_size=100)
address = tf.feature_column.categorical_column_with_hash_bucket('address',hash_bucket_size=100)
buildingTypeName = tf.feature_column.categorical_column_with_hash_bucket('buildingTypeName',hash_bucket_size=100)

#经纬度：
longitude_bucket = tf.feature_column.bucketized_column(longitude,[-100,-60,-30,-10])
latitude_bucket = tf.feature_column.bucketized_column(latitude,[40,43,46,49,52,55])


# 对年月日进行分桶操作：其中对年进行categorical_column 操作，对月份和号数进行buckized_column 操作
# 对于year 其实可以采用tf.feature_column.categorical_column_with_identity 来表示，他是分桶的一种特殊情况
# 分桶表示的是一个范围，而他是表示一个唯一的整数；他是分类标识列；

year_categorical = tf.feature_column.categorical_column_with_vocabulary_list('year',['2017','2018'])
month_bucket = tf.feature_column.bucketized_column(month,[4,7,10])
day_bucket = tf.feature_column.bucketized_column(day,[11,21])



# 定义基本特征和组合特征
base_columns = [
 price_bucket,expectedDealPrice_bucket, tradeTypeName, province, city, address,buildingTypeName, year_categorical,
    month_bucket, day_bucket,longitude_bucket,latitude_bucket
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
    # expectedDealPrice,
    # embedding将高纬的稀疏tensor转化成低维的tensor
    tf.feature_column.embedding_column(province,8),
    tf.feature_column.embedding_column(city,8),
    tf.feature_column.embedding_column(address,8),
    tf.feature_column.embedding_column(buildingTypeName,8),

    tf.feature_column.indicator_column(tradeTypeName),
    tf.feature_column.indicator_column(year_categorical),


    # tf.feature_column.indicator_column(province),
    # tf.feature_column.indicator_column(city),
    # tf.feature_column.indicator_column(address),
    # tf.feature_column.indicator_column(buildingTypeName)
]

# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./tmp_official3/predict_model',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns = deep_columns,
    # dnn_hidden_units= [1024, 512, 256,128,64,32,16],
    dnn_hidden_units=[16,32,64,128,256,512,1024,2048]
    # linear_optimizer=tf.train.AdadeltaOptimizer(), # 对于稀疏的数据用自适应优化器更好；
    # dnn_optimizer= tf.train.AdamOptimizer()
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
        'buildingTypeName': np.array(example['buildingTypeName']),
        'tradeTypeName': np.array(example['tradeTypeName']),
        'expectedDealPrice': np.array(example['expectedDealPrice']),
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
        'buildingTypeName': np.array(example_test['buildingTypeName']),
        'tradeTypeName': np.array(example_test['tradeTypeName']),
        'expectedDealPrice': np.array(example_test['expectedDealPrice']),
        'year': np.array(test_date_data['year']).astype('str'),
        'month': np.array(test_date_data['month']),
        'day': np.array(test_date_data['day'])
       },
    y=np.array(label_test),
    num_epochs=1, # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False
)











# 训练
for j in range(200):
    estimator_model.train(input_fn=train_input_fn,steps=1000)

# estimator_model.train(input_fn=train_input_fn,steps=1000)

# 测试
ev = estimator_model.evaluate(input_fn=test_input_fn, steps=1)
print('ev: {}'.format(ev))
# 预测
predict_iter = estimator_model.predict(input_fn=test_input_fn)

# predict_iter_list = [x for x in predict_iter]

# 循环次数根据测试数据数决定
list_value =[]
for i in range(27):
    #目前不知道这个dict_values([array([51.575745], dtype=float32)]) 数据怎么把值弄出来，就没有算精确度了
    x = float(list(predict_iter.__next__().values())[0])
    print(i, x)
    list_value.append(x)

print(list_value)
print(mean_absolute_error(label_test, list_value))