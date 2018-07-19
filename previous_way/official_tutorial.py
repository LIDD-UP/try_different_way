#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: official_tutorial.py
@time: 2018/7/3
"""

'''

处理方式有对于省份和城市还有地址采用分桶的策略，
对于连续的值：经度，维度，价格，和期望价格采用普通的方式，就用feature_column.numeric_column('price')进行；但是要不要考虑对这些数据进行分桶以提高准确率；也就是使用bucketized_column来实现；
对于分类数据：就用tf.feature_column.categorical_column_with_vocabulary_list(tradeTypeName,[Sales,Lease])来进行，作为线性模型的输入，或者是DNN模型输入（单数在输入之前需要将他转化成密集型的）通过indicator_column进行转换；
最后就是需不需要将默写特征作为嵌入型的列；

宽模型的特征是稀疏的和交叉的特征组成的；

具有交叉特征列的宽模型可以有效地记忆特征之间的稀疏交互。
话虽如此，交叉特征列的一个限制是它们没有概括为未出现在训练数据中的特征组合。让我们添加一个带有嵌入的深层模型来修复它。

深度模型：嵌入式神经网络
深度模型是前馈神经网络，如上图所示。
首先将每个稀疏的高维分类特征转换成低维且密集的实值向量，通常称为嵌入向量。这些低维密集嵌入向量与连续特征连接，
然后在前向传递中馈入神经网络的隐藏层。嵌入值随机初始化，并与所有其他模型参数一起训练，以最小化训练损失。
 
 multi-hot ，one_hot编码 也就是一个是0，1，另一个则是有其他数字；

而深度模型则是使用的


此处应该注意，是否应该对如果某一条数据空了，是否就应该舍弃该数据；
对日期的处理应该是如果为空，不出处理，如果不为空，则应该根据月份来实现；

'''
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



# 这里header 是零就可以了，
data = pd.read_csv(dirname+train_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,12,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'listingDate', 'daysOnMarket'])  # header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
# print(data[['longitude']]) #他是一个二位的数组，需要有两个【】这个；
data = data.dropna(axis=0)
example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice','listingDate']]
print(type(example))
print(example['listingDate'])
label = data[['daysOnMarket']]



# 加载测试数据
# data_test = pd.read_csv(dirname+test_filename,header=1,usecols=[4,5,6,11,14] ,names=['longitude','latitude','price','expectedDealPrice','daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
data_test = pd.read_csv(dirname+test_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'daysOnMarket'])  #header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据

data_test = data_test.dropna(axis=0)
example_test = data_test[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice']]
label_test = data_test[['daysOnMarket']]

print(example, label_test)






# longitude	latitude price	buildingTypeId	buildingTypeName	tradeTypeId	tradeTypeName
# expectedDealPrice
# listingDate	delislingDate	daysOnMarket

# 连续
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
price = tf.feature_column.numeric_column('price')  # 可以考虑使用分桶策略
expectedDealPrice =tf.feature_column.numeric_column('expectedDealPrice')  # 可以考虑使用分桶策略

tradeTypeName = tf.feature_column.categorical_column_with_vocabulary_list('tradeTypeName', ['Sale', 'Lease'])


# 对于这种种类过多的情况，我们也可以使用categorical_column_with_hash_bucket  来实现：
# buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId',['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'],default_value=-1)

# 对于省份，城市，地址这种种类过多的情况我们采用 ，categorical_column_with_hash_bucket 这种方式；
province = tf.feature_column.categorical_column_with_hash_bucket('province',hash_bucket_size=1000)
city = tf.feature_column.categorical_column_with_hash_bucket('city',hash_bucket_size=1000)
address = tf.feature_column.categorical_column_with_hash_bucket('address',hash_bucket_size=1000)
buildingTypeName = tf.feature_column.categorical_column_with_hash_bucket('buildingTypeName',hash_bucket_size=1000)

# 定义基本特征和组合特征
base_columns = [
 tradeTypeName, province, city, address,buildingTypeName
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
    tf.feature_column.embedding_column(province,8),
    tf.feature_column.embedding_column(city,8),
    tf.feature_column.embedding_column(address,8),
    tf.feature_column.embedding_column(buildingTypeName,8),

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
    dnn_hidden_units= [1024, 512, 256,128,64,32,16],
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






# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={
#         'province': np.array(['四川','黑龙江']),
#         'city': np.array(['成都','哈尔冰']),
#         'address': np.array(['悉尼','墨尔本']),
#         'longitude': np.array([-123,56]),
#         'latitude': np.array([123,12]),
#         'price': np.array([55555,555565]),
#         # 'buildingTypeId': np.array(example_test['buildingTypeId']),
#         'tradeTypeName': np.array(['Sale','Lease']),
#         'expectedDealPrice': np.array([555565,666666])
#        },
#     y=np.array([16,21]),
#     num_epochs=None,
#     shuffle=True
# )




# 训练
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





# for predict in predict_iter:
#     print(predict['predictions'])

