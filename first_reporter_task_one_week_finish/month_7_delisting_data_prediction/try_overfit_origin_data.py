# -*- coding:utf-8 _*-
"""
@author:Administrator
@file: standard_to_predict.py
@time: 2018/9/18
"""
import tensorflow as tf
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import sys
from sklearn.preprocessing import MinMaxScaler,StandardScaler


root_path = "F:\\PycharmProject\\try_different_way\\"
dir_path = os.path.dirname(root_path + "first_reporter_task_one_week_finish/month_7_delisting_data_prediction/savemodel_overfit/")

# 日志
# tf.logging.set_verbosity(tf.logging.INFO)


# preprocess data
def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        "city",
        "province",
        "price",
        "tradeTypeId",
        # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
    ]]

    data = data.dropna(axis=0)
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list
    data = data.dropna(axis=0)
    return data

# 归一化数据：

def normalization_price(data):
    minmax = StandardScaler()
    data_price= np.array(data[['longitude','price','latitude',
                               # 'bedrooms','bathroomTotal'
                               ]])
    data[['longitude', 'price', 'latitude',
          # 'bedrooms','bathroomTotal'
          ]] = minmax.fit_transform(data_price)
    return data


# 对label进行log变换；
def log_transform(data):
    data = np.log1p(data)
    return data





# 加载训练数据
data = pd.read_csv('./input/month_7_train_after_process_1.csv')
data = preprocess_data(data)

data= normalization_price(data)
print(data.head())

example = data[['province', 'city', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
                # 'listingDate',
                'bedrooms','bathroomTotal']]

label = data[['daysOnMarket']]
label = log_transform(label)




# 加载测试数据
data_test = pd.read_csv('../prediction_data/month_8_data_after_process_1.csv')
data_test = preprocess_data(data_test)

origin_data = data_test.reset_index()


data_test = normalization_price(data_test)

example_test = data_test[
    ['province', 'city', 'longitude', 'latitude', 'price', 'buildingTypeId', 'tradeTypeId',
      # 'listingDate',
     'bedrooms','bathroomTotal']]
label_test = data_test[['daysOnMarket']]





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
# example_date_data = date_processing(example)
# test_date_data = date_processing(example_test)


# 生成经纬度
def generate_longtitude_and_latitude_list(min,max,distance):
    list_len = (max -min)/distance
    list_boundaries =[]
    middle = min
    for i in range(int(list_len)):
        middle += distance
        list_boundaries.append(middle)
    return list_boundaries
# longitude_boudaries = generate_longtitude_and_latitude_list(longitude_min, longitude_max, 0.005)
# latitude_boudaries = generate_longtitude_and_latitude_list(latitude_min, latitude_max, 0.005)



# 变化的-------------------------------------------------------------》》》》》》》》》》》》》》》




# city = tf.feature_column.categorical_column_with_hash_bucket('city', hash_bucket_size=1000)
# address = tf.feature_column.categorical_column_with_hash_bucket('address', hash_bucket_size=100)


longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
# longitude_bucket = tf.feature_column.bucketized_column(longitude, sorted(longitude_boudaries))
# latitude_bucket = tf.feature_column.bucketized_column(latitude, sorted(latitude_boudaries))
# longitude_latitude = tf.feature_column.crossed_column(
#         [longitude_bucket, latitude_bucket], 1000
#     )


def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i + 1.) / (num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]



price = tf.feature_column.numeric_column('price')
# price_bucket = tf.feature_column.bucketized_column(price,
#                                                    [500000, 1000000, 1500000,
#                                                     2000000, 4000000])
# price_bucket = tf.feature_column.bucketized_column(
#     price,
#     boundaries=get_quantile_based_buckets(example["price"], 80))

buildingTypeId = tf.feature_column.categorical_column_with_vocabulary_list('buildingTypeId',
                list(set(list(set(data['buildingTypeId'])) + list(set(data_test['buildingTypeId']))))
                                                                                )
province = tf.feature_column.categorical_column_with_vocabulary_list('province',
                set(list(set(data['province'])) + list(set(data_test['province'])))
                                                                     )

city = tf.feature_column.categorical_column_with_vocabulary_list('city',
                set(list(set(data['city'])) + list(set(data_test['city'])))
                                                                     )

#
# month = tf.feature_column.numeric_column('month')
# day = tf.feature_column.numeric_column('day')
# month_bucket = tf.feature_column.categorical_column_with_vocabulary_list('month', [1,2,3,4,5,6,7,8,9,10,11,12])
# day_bucket = tf.feature_column.bucketized_column(day, [11, 21])
#
#
bedrooms = tf.feature_column.numeric_column('bedrooms')
bathroomTotal = tf.feature_column.numeric_column('bathroomTotal')



deep_columns = [

    # tf.feature_column.embedding_column(city, 8),
    # tf.feature_column.embedding_column(address, 8),

    latitude,
    longitude,
    # tf.feature_column.embedding_column(longitude_latitude, 8),

    price,
    # price_bucket,

    tf.feature_column.indicator_column(buildingTypeId),
    tf.feature_column.indicator_column(province),
    tf.feature_column.indicator_column(city),

    # day_bucket,
    # tf.feature_column.indicator_column(month_bucket),

    bedrooms,
    bathroomTotal,
]













# 以下都是不变的-----------------------------------------------------------》》》》》》》》》》》》》》》》》

# 定义模型（估计器）
dnn_regressor = tf.estimator.DNNRegressor(
    model_dir=dir_path,
    feature_columns=deep_columns,
    hidden_units=[512, 256, 128, 64, 32],
    # dnn_optimizer=tf.train.AdamOptimizer()
)


def train_input_fn(is_shuffe=True,batch_size=100,num_epoch=None):
     return tf.estimator.inputs.numpy_input_fn(
        x={
            'province': np.array(example['province']),
            'city': np.array(example['city']),
            # 'address': np.array(example['address']),
            'longitude': np.array(example['longitude']),
            'latitude': np.array(example['latitude']),
            'price': np.array(example['price']),
            'buildingTypeId': np.array(example['buildingTypeId']).astype('int'),
            # 'buildingTypeId': np.array(example['buildingTypeId']).astype('str'),
            # 'month': np.array(example_date_data['month']).astype('int'),
            # 'day': np.array(example_date_data['day']),
            'bedrooms': np.array(example['bedrooms']),
            'bathroomTotal': np.array(example['bathroomTotal']),
        },
        y=np.array(label),
        num_epochs=num_epoch,
        shuffle=is_shuffe,
        batch_size=batch_size,
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
        # 'buildingTypeId': np.array(example_test['buildingTypeId']).astype('str'),
        # 'month': np.array(test_date_data['month']).astype('int'),
        # 'day': np.array(test_date_data['day']),
        'bedrooms': np.array(example_test['bedrooms']),
        'bathroomTotal': np.array(example_test['bathroomTotal']),
    },
    y=np.array(label_test),
    num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
    shuffle=False,
    batch_size=100,
)




def train_and_validation_function():
    periods = 10
    training_rmse = []
    validation_rmse = []
    for period in range(periods):

        steps_trains = int(len(example) / 10)
        print(steps_trains)
        dnn_regressor.train(input_fn=train_input_fn(), steps=steps_trains)
        training_predictions = dnn_regressor.predict(input_fn=train_input_fn(is_shuffe=False,batch_size=1000,num_epoch=1))
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])


        validation_predictions = dnn_regressor.predict(input_fn=test_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        validation_predictions = np.expm1(validation_predictions)

        # export validation data to csv file
        predictions = pd.DataFrame()
        predictions['predictions'] = validation_predictions
        predictions['predictions'] = round(abs(predictions['predictions']))


        print(predictions.describe())
        predictions.to_csv('./validation_predict_result_overfit.csv', index=False)
        merge_data = pd.concat((origin_data, predictions), axis=1)

        merge_data_df = pd.DataFrame(merge_data)
        merge_data_df.to_csv('./validation_merge_result_overfit.csv', index=False)


        # Compute training and validation loss.
        training_root_mean_squared_error = metrics.mean_absolute_error(training_predictions, label)
        validation_root_mean_squared_error = metrics.mean_absolute_error(validation_predictions, label_test)
        # Occasionally print the current loss.
        print(" train-error: period %02d : %0.2f" % (period, training_root_mean_squared_error))
        print(" vaidation-error period %02d : %0.2f" % (period, validation_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        # # if training_root_mean_squared_error <= 80:
        #     break
        period += 1

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

train_and_validation_function()


