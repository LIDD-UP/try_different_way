from __future__ import print_function

import tensorflow as tf
import os as os
import numpy as np
import pandas as pd
from sklearn import metrics
from IPython import display
from tensorflow.python.data import Dataset

from matplotlib import pyplot as plt

# 根目录
root_path = "E:/Workspaces/Python/Tensorflow/"

# Model存储路径
dir_path = os.path.dirname(root_path + "daysOnMarket/Model2018/")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 30
pd.options.display.float_format = '{:.1f}'.format

canada_housing_data = pd.read_csv(root_path + "house_info_2018.csv", sep=",")

canada_housing_data = canada_housing_data.dropna(axis=0)

canada_housing_data = canada_housing_data[canada_housing_data['daysOnMarket'] <= 365]
canada_housing_data = canada_housing_data[canada_housing_data['longitude'] != 0]
canada_housing_data = canada_housing_data[canada_housing_data['latitude'] != 0]
canada_housing_data = canada_housing_data[canada_housing_data['price'] > 1]

canada_housing_data = canada_housing_data.reindex(
        np.random.permutation(canada_housing_data.index))
canada_housing_data = (canada_housing_data.head(250000)).copy()
# canada_housing_data = canada_housing_data.fillna(canada_housing_data.mean())

# 测试数据
canada_housing_test_data = pd.read_csv(root_path + "house_info_test.csv", sep=",")

canada_housing_test_data = canada_housing_test_data[canada_housing_test_data['daysOnMarket'] <= 365]


def pre_process_targets(canada_housing_data):
    output_targets = pd.DataFrame()
    output_targets["daysOnMarket"] = canada_housing_data["daysOnMarket"]

    return output_targets

def pre_process_features(canada_housing_data):
    selected_features = canada_housing_data[
        ["longitude",
         "latitude",
         "price",
         "tradeTypeId",
         # "listingDate",
         "buildingTypeId"
         ]]
    # data_lenth = len(canada_housing_data)
    # processed_features = (selected_features.head(int(data_lenth*0.5))).copy()
    processed_features = selected_features.copy()
    processed_features["longitude"] = round(processed_features["longitude"], 2)
    processed_features["latitude"] = round(processed_features["latitude"], 2)
    listingDateMonth = []
    for item in canada_housing_data["listingDate"]:
        listingDateMonth.append(int(item.split('/')[1]))
    processed_features["listingDataMonth"] = listingDateMonth

    return processed_features


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(training_examples):
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """
    # numeric_trade_type_id = tf.feature_column.numeric_column("tradeTypeId")
    building_type_vocabulary_list=[]
    for i in range(training_examples["buildingTypeId"].max()):
        building_type_vocabulary_list.append(i+1)
    vocabulary_building_type_id = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="buildingTypeId",
            vocabulary_list=building_type_vocabulary_list))

    vocabulary_listing_data_month = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="listingDataMonth",
            vocabulary_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

    vocabulary_trade_type_id = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="tradeTypeId",
            vocabulary_list=[1, 2]))
    # numeric_price = tf.feature_column.numeric_column("price")
    bucketized_price = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("price"),
        boundaries=get_quantile_based_buckets(training_examples["price"], 100))
    bucketized_longitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("longitude"),
        boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
    bucketized_latitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("latitude"),
        boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))

    long_x_lat = tf.feature_column.embedding_column(
        tf.feature_column.crossed_column(
            keys=[bucketized_longitude, bucketized_latitude], hash_bucket_size=1000), 8)

    feature_columns = [
        # vocabulary_listing_data_month,
        vocabulary_building_type_id,
        # numeric_price,
        bucketized_price,
        vocabulary_trade_type_id,
        bucketized_longitude,
        bucketized_latitude,
        long_x_lat]

    return feature_columns


def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i + 1.) / (num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]

def create_dnn_regressor(hidden_units, training_examples):
    # 优化函数，后续修改, DNNRegressor默认使用Adagrad
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        model_dir=dir_path,
        hidden_units=hidden_units
    )
    return dnn_regressor


def train_nn_regression_model(
        steps,
        batch_size,
        canada_housing_data):

    # 训练数据百分比
    TRAINING_PERCENT = 0.8
    # 训练次数
    periods = 10
    steps_per_period = steps / periods

    data_lenth = len(canada_housing_data)

    # 训练数据
    # training_data = (canada_housing_data.head(int(data_lenth * TRAINING_PERCENT))).copy()
    training_data = (canada_housing_data.head(250000)).copy()
    # 验证数据
    validate_data = (training_data.tail(10000)).copy()

    training_examples = pre_process_features(training_data)
    training_targets = pre_process_targets(training_data)

    validation_examples = pre_process_features(validate_data)
    validation_targets = pre_process_targets(validate_data)
    dnn_regressor = create_dnn_regressor(
        hidden_units=[512, 256, 128, 64, 32],
        training_examples=training_examples)
    # 数据检查.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print("Validation examples summary:")
    display.display(validation_examples.describe())

    print("Training targets summary:")
    display.display(training_targets.describe())
    print("Validation targets summary:")
    display.display(validation_targets.describe())

    training_input_fn = lambda: my_input_fn(
        features=training_examples,
        targets=training_targets["daysOnMarket"],
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    predict_training_input_fn = lambda: my_input_fn(
        features=training_examples,
        targets=training_targets["daysOnMarket"],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
        features=validation_examples,
        targets=validation_targets["daysOnMarket"],
        num_epochs=1,
        shuffle=False)
    # predict_training_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(training_examples)},
    #     y=np.array(training_targets["daysOnMarket"]),
    #     num_epochs=1,
    #     shuffle=False)
    # predict_validation_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(validation_examples)},
    #     y=np.array(validation_targets["daysOnMarket"]),
    #     num_epochs=1,
    #     shuffle=False)
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.

        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = metrics.mean_absolute_error(training_predictions, training_targets)
        validation_root_mean_squared_error = metrics.mean_absolute_error(validation_predictions, validation_targets)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        # # if training_root_mean_squared_error <= 80:
        #     break

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

    return dnn_regressor


def test_predict_nn_regression_model(dnn_regressor, canada_housing_test_data):

    test_examples = pre_process_features(canada_housing_test_data)
    test_targets = pre_process_targets(canada_housing_test_data)

    test_predictions = dnn_regressor.predict(
        input_fn=lambda: my_input_fn(
            features=test_examples,
            targets=test_targets["daysOnMarket"],
            num_epochs=1,
            shuffle=False))

    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = metrics.mean_absolute_error(test_predictions, test_targets)
    print("Final RMSE (on testing data): %0.2f" % root_mean_squared_error)

    total_DOM = 0
    for item in test_targets['daysOnMarket']:
        total_DOM += item

    display.display("测试数据DOM平均天数 %0.2f" % (total_DOM / len(test_targets['daysOnMarket'])))

    result = 0

    for index in range(len(test_predictions)):
        result += test_predictions[index]

    display.display("测试数据据测结果平均天数 %0.2f" % (result / len(test_predictions)))


if __name__ == "__main__":

    # test_examples = pre_process_features(canada_housing_data)

    # 训练
    dnn_regressor = train_nn_regression_model(
        steps=5000,
        batch_size=50,
        canada_housing_data=canada_housing_data)

    # feature_spec = {"listingDataMonth": tf.placeholder(shape=[None], dtype=tf.int64),
    #                 "longitude": tf.placeholder(shape=[None], dtype=tf.string),
    #                 "latitude": tf.placeholder(shape=[None], dtype=tf.string),
    #                 "price": tf.placeholder(shape=[None], dtype=tf.float32),
    #                 "tradeTypeId": tf.placeholder(shape=[None], dtype=tf.int64),
    #                 "buildingTypeId": tf.placeholder(shape=[None], dtype=tf.int64)}
    #
    # serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    #
    # dnn_regressor.export_savedmodel(dir_path, serving_input_receiver_fn)
    # 测试
    canada_housing_data = canada_housing_data.reindex(
        np.random.permutation(canada_housing_data.index))
    test_predict_nn_regression_model(
        dnn_regressor=dnn_regressor,
        canada_housing_test_data=(canada_housing_data.head(10000)).copy())
