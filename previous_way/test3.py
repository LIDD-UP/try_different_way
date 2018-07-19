import pandas as pd
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os



dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator_new\\house_info.csv'
test_filename = '\\use_estimator_new\\test_house_info.csv'

house_data = pd.read_csv(dirname + train_filename)

data = house_data.dropna()
data = data.drop(['province'],1)
data = data.drop(['city'],1)
data = data.drop(['address'],1)
data = data.drop(['postalCode'],1)
data = data.drop(['buildingTypeName'],1)
data = data.drop(['tradeTypeName'],1)
data = data.drop(['listingDate'],1)
data = data.drop(['delislingDate'],1)

data=data[data['longitude'] < 0]
data=data[data['longitude'] > -134.485]
data=data[data['latitude'] < 57.9533]
data=data[data['latitude'] > 35.9605]
data=data[data['price'] <= 1205540]
data=data[data['expectedDealPrice'] <= 1194230]
data=data[data['daysOnMarket'] <= 133]

# data = data.drop(['price'],1)
# data = data.drop(['expectedDealPrice'],1)

#用Sigmoid函数实现离散值归一化
# data['buildingTypeId'] = 1.0 / (1 + np.exp(-data['buildingTypeId']))
# data['tradeTypeId'] = 1.0 / (1 + np.exp(-data['tradeTypeId']))

#Min-Max Normalization
data['price'] = abs((data['price']-np.min(data['price']))/(np.max(data['price'])-np.min(data['price'])))
data['expectedDealPrice'] = abs((data['expectedDealPrice']-np.min(data['expectedDealPrice']))
                                /(np.max(data['expectedDealPrice'])-np.min(data['expectedDealPrice'])))

"""
def latLng2WebMercator(lng,lat):
    lng = np.array(lng)
    lat = np.array(lat)
    earthRad = 6378137.0
    x = lng * math.pi / 180 * earthRad
    for i in lat:
        a = i * math.pi / 180
        y = earthRad / 2 * math.log((1.0 + math.sin(a)) / (1.0 - math.sin(a)))
    return x, y

data['longitude'],data['latitude']=latLng2WebMercator(data['longitude'],data['latitude'])
print(data['longitude'],data['latitude'])
"""
data['longitude'] = (data['longitude']-np.min(data['longitude']))/(np.max(data['longitude'])-np.min(data['longitude']))
data['latitude'] = (data['latitude']-np.min(data['latitude']))/(np.max(data['latitude'])-np.min(data['latitude']))

# plt.figure()
# p = data.boxplot()
# plt.show()

#print(data)

FEATURES = ["longitude","latitude","price","buildingTypeId","tradeTypeId","expectedDealPrice"]
LABEL = "daysOnMarket"

training_data = data.ix[:40500,:6]
training_label = data.ix[:40500,6]
testing_data = data.ix[40501:51000,:6]
testing_label = data.ix[40501:51000,6]

# training_data = data.ix[:295,:6]
# training_label = data.ix[:295,6]
# testing_data = data.ix[296:300,:6]
# testing_label = data.ix[296:300,6]


# ## 定义 FeatureColumns
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# ## 定义 regressor
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10,128])

# ## 定义 input_fn
def input_fn(df, label):
    feature_cols = {k: tf.constant(df[k]) for k in FEATURES}
    label = tf.constant(label)
    return feature_cols, label


def train_input_fn():
    '''训练阶段使用的 input_fn'''
    return input_fn(training_data, training_label)


def test_input_fn():
    '''测试阶段使用的 input_fn'''
    return input_fn(testing_data, testing_label)

# 训练
regressor.fit(input_fn=train_input_fn, steps=1000)
# 测试
ev = regressor.evaluate(input_fn=train_input_fn, steps=1)
print('ev: {}'.format(ev))

predictions = list(regressor.predict(input_fn=test_input_fn))
for index in range(len(predictions)):
    if predictions[index] < 0:
        predictions[index] = 0
print("误差：",sum(abs(predictions - testing_label))/len(testing_label))
for i in range(len(testing_label)):
    print("true label:",list(testing_label)[i])
    print("Predictions: {}".format(str(predictions[i])))
    
