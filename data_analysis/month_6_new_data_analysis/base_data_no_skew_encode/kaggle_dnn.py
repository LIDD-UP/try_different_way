# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: kaggle_dnn.py
@time: 2018/9/5
"""

import pandas as pd

pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np  # linear algebra
from matplotlib import pyplot as plt  # for plotting graphs
from functools import cmp_to_key
from sklearn import metrics
import math
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm, skew

tf.logging.set_verbosity(tf.logging.ERROR)

'''

'''

data = pd.read_csv('./base_data_no_skew_encode.csv')
data['buildingTypeId'] = data['buildingTypeId'].astype('str')
print(data.head())
data = data.drop(columns=[
    'province',
    'city',
    'address',
    'postalCode',
    'tradeTypeId',
])
data = data.dropna()


_x = data.drop(columns=['daysOnMarket'])
_y = data['daysOnMarket']

# 数据处理?\feature 需要用到skew；

print(_x.head())
# def log_transform_data(data):
#     for column in data.columns:
_x['longitude'] = abs(_x['longitude'])
# _x['buildingTypeId'] = _x['buildingTypeId'].astype('str')
def get_process_skew_numeric_feature(data):
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)

    # 将处理skew的特征
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # data[feat] += 1
        data[feat] = boxcox1p(data[feat], lam)

    # data[skewed_features] = np.log1p(data[skewed_features])
    return data


_x = get_process_skew_numeric_feature(_x)


def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            data[column] = data[column].astype('str')
    return data


_x = label_encode(_x)

# dummies
def dummies_class_variable(data):
    data = pd.get_dummies(data)
    print(data.shape)
    return data


_x = dummies_class_variable(_x)
print('dummies successful')

all_data = pd.concat((_x,_y),axis=1)

# 将数据拆分为训练和测试数据用train_test_split,用它存在每次的数据都会有变化，会将训练的数据拿来测试
# 所以还是直接分成测试数据1000，其余的都是训练数据

df  = all_data.iloc[1000:,:]
df_test = all_data.iloc[0:1000,:].drop(columns='daysOnMarket')
df_test_label = all_data.iloc[0:1000,:]['daysOnMarket']

print(df_test.shape, df.shape)
print(df.head())

# check thess vareables
numerical_fields = [

    numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
]

categorical_fields = [
    categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
]

fields = numerical_fields + categorical_fields


# Neural Network
# Use un-processed numerical features only to see how good can it be.

def log2(n):
    if n <= 0:
        return 0.0
    else:
        return math.log(n)


def get_logged_series(s):
    s1 = [log2(n) for n in s]
    return s1


def get_logged_df(df):
    df1 = pd.DataFrame()
    for k in df.keys():
        s1 = get_logged_series(s)
        df1[k] = s1
    return df1


def get_scaled_series(base=1.0):
    def wrapper_get_scaled_series(s):
        s_max = max(s)
        return [base * float(n) / float(s_max) for n in s]

    return wrapper_get_scaled_series


def get_scaled_df(df, base=1.0):
    df1 = pd.DataFrame()
    for k in df.keys():
        s = df[k]
        s1 = get_scaled_series(base=base)(s)
        df1[k] = s1
    return df1


def get_dummies(dummy_na=False):
    def wrapper_get_dummies(s):
        df = pd.get_dummies(s, prefix=s.name, dummy_na=dummy_na)
        df1 = pd.DataFrame()
        for k in df.keys():
            s = df[k]
            name = s.name
            name = name.replace('(', '')
            name = name.replace(')', '')
            name = name.replace(' ', '')
            s1 = s.rename(name)
            df1[s1.name] = s1
        return df1

    return wrapper_get_dummies


# 其实就是数据处理的过程，分成分类型数据和连续性数据
def convert_features(df, num_fields, cat_fields, num_fields_proc=None, cat_fields_proc=None, label_name=None,
                     train_validate_ratio=None):
    if num_fields_proc is None:
        num_fields_proc = lambda x: x
    if cat_fields_proc is None:
        cat_fields_proc = lambda x: x
    features = pd.DataFrame()
    for k in num_fields:
        features[k] = num_fields_proc(df[k].copy())
    for k in cat_fields:
        features = features.join(cat_fields_proc(df[k].copy()))
    if label_name is not None:
        labels = df[label_name].copy()
    else:
        labels = None
    if train_validate_ratio is None:
        train_validate_ratio = 1
    train_num = int(len(df) * train_validate_ratio)
    validate_num = len(df) - train_num
    train_features = features.head(train_num)
    validate_features = features.tail(validate_num)
    if labels is not None:
        train_labels = labels.head(train_num)
        validate_labels = labels.tail(validate_num)
    else:
        train_labels = None
        validate_labels = None
    return (train_features, train_labels, validate_features, validate_labels)


# 特征工程
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
    input_features: The names of the numerical input features to use.
    Returns:
    A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


# 构建输入
def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    # features = {key:np.array(value) for key,value in dict(features).items()}

    #
    # note: can convert to dict directly
    #
    features = dict(features)

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 预测的输入
def my_input_fn_pred(features, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    # features = {key:np.array(value) for key,value in dict(features).items()}

    #
    # note: can convert to dict directly
    #
    features = dict(features)

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices(features)  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features = ds.make_one_shot_iterator().get_next()
    return features


# 构建模型
def train_dnn_regressor_model(
        optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a DNN regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

    Returns:
    A `LinearRegressor` object trained on the training data.
    """
    # RMSE, RMSLE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    if validation_examples is not None and validation_targets is not None:
        do_validation = True
    else:
        do_validation = False

    # 训练多少次；
    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    dnn_regressor = tf.estimator.DNNRegressor(
        model_dir='./file_dnn',
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=optimizer
    )

    # Create input functions.
    # 分成三份数据，训练，验证和测试
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets,
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets,
                                                    num_epochs=1,
                                                    shuffle=False)
    if do_validation == True:
        predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                          validation_targets,
                                                          num_epochs=1,
                                                          shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE, RMSLE (on training data):")
    training_rmse = []
    validation_rmse = []
    training_rmsle = []
    validation_rmsle = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        # 用相同的数据做预测计算loss值
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        # Compute training loss RMSE.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_targets, training_predictions))

        # Compute training loss RMSLE.
        training_root_mean_squared_log_error = math.sqrt(
            metrics.mean_squared_log_error(training_targets, training_predictions))

        # 当把数据集切分的时候，就做验证；
        if do_validation == True:
            validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

            # Compute validation loss RMSE.
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_targets, validation_predictions))

            # Compute validation loss RMSLE.
            validation_root_mean_squared_log_error = math.sqrt(
                metrics.mean_squared_log_error(validation_targets, validation_predictions))

        # Occasionally print the current loss.
        print("  period %02d : %0.2f, %0.4f" % (
            period, training_root_mean_squared_error, training_root_mean_squared_log_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        training_rmsle.append(training_root_mean_squared_log_error)
        if do_validation == True:
            validation_rmse.append(validation_root_mean_squared_error)
            validation_rmsle.append(validation_root_mean_squared_log_error)

    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.figure(figsize=(15, 5))
    # RMSE
    plt.subplot(1, 3, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(training_rmse, label="training")
    if do_validation == True:
        plt.plot(validation_rmse, label="validation")
    plt.legend()
    # RMSLE
    plt.subplot(1, 3, 2)
    plt.ylabel("RMSLE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Logarithmic Error vs. Periods")
    plt.plot(training_rmsle, label="training")
    if do_validation == True:
        plt.plot(validation_rmsle, label="validation")
    plt.legend()
    # Target / Prediction
    plt.subplot(1, 3, 3)
    plt.ylabel("Target")
    plt.xlabel("Prediction")
    plt.title("Target vs. Prediction")
    lim = max(training_targets)
    if do_validation == True:
        lim = max(lim, max(validation_targets))
    lim *= 1.05
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], alpha=0.5, color='red')
    plt.scatter(training_predictions, training_targets, alpha=0.5, label="training")
    if do_validation == True:
        plt.scatter(validation_predictions, validation_targets, alpha=0.5, label="validation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return dnn_regressor


def no_process_train_raw_data():
    # no process raw data
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=None,
                           # num_fields_proc=get_scaled_series(base=1000.0),
                           # num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='daysOnMarket',
                           train_validate_ratio=0.7)

    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=2000,
        batch_size=100,
        hidden_units=[22, 44, 22, 11],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    return dnn_regressor


# Training model...


DNN_model = no_process_train_raw_data()


def scale_to_1000():
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_scaled_series(base=1000.0),
                           cat_fields_proc=None,
                           label_name='daysOnMarket',
                           train_validate_ratio=0.7)

    # Scale all features to 0 ~ 1000
    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=2000,
        batch_size=100,
        hidden_units=[22, 44, 22, 11],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    # Training model...
    return dnn_regressor


# DNN_model_scale = scale_to_1000()


def log_transform():
    # Try scale features with log
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='SalePrice',
                           train_validate_ratio=0.7)

    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=2000,
        batch_size=100,
        hidden_units=[22, 44, 22, 11],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    return dnn_regressor


# DNN_model_log = log_transform()


def change_hidden_layer_setting():
    # Change hidden layer settings
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='SalePrice',
                           train_validate_ratio=0.7)

    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=400000,
        batch_size=2,
        hidden_units=[11, 12, 13, 12, 11, 3],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    return dnn_regressor


# DNN_model_change_hidden_layer = change_hidden_layer_setting()


def use_adagradoptimaizer():
    # Use AdagradOptimizer with more iterations, change batch to 10
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='SalePrice',
                           train_validate_ratio=0.7)

    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]
    # Use AdagradOptimizer with more iterations, change batch to 10

    optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=1000000,
        batch_size=10,
        hidden_units=[11, 12, 13, 12, 11, 3],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    return dnn_regressor


# DNN_model_use_adagradoptimaizer = use_adagradoptimaizer()
# ------------------------------------------------------------------------------->>>>>>
def add_category_feature():
    # add category feature
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='SalePrice',
                           train_validate_ratio=0.7)

    training_features = ret[0]
    training_targets = ret[1]
    validation_features = ret[2]
    validation_targets = ret[3]

    optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=100000,
        batch_size=100,
        hidden_units=[11, 12, 13, 12, 11, 3],
        training_examples=training_features,
        training_targets=training_targets,
        validation_examples=validation_features,
        validation_targets=validation_targets)
    return dnn_regressor


# DNN_model_add_category = add_category_feature()

def new1():
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]

    ret = convert_features(df,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name='SalePrice',
                           train_validate_ratio=1.0)

    training_features_all = ret[0]
    training_targets_all = ret[1]

    optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = train_dnn_regressor_model(
        optimizer,
        steps=100000,
        batch_size=100,
        hidden_units=[11, 12, 13, 12, 11, 3],
        training_examples=training_features_all,
        training_targets=training_targets_all,
        validation_examples=None,
        validation_targets=None)
    return dnn_regressor


# DNN_model_new = new1()


# 预测；
def predict_test(dnn_regressor):
    numerical_fields = [
        numeric_column for numeric_column in df_test.columns if df_test[numeric_column].dtype != 'object'
    ]

    categorical_fields = [
        categorical_column for categorical_column in df_test.columns if df_test[categorical_column].dtype == 'object'
    ]
    ret = convert_features(df_test,
                           numerical_fields,
                           categorical_fields,
                           num_fields_proc=get_logged_series,
                           cat_fields_proc=None,
                           label_name=None,
                           train_validate_ratio=1.0)

    test_features = ret[0]

    # haneld nan fields
    for k in test_features.keys():
        s = test_features[k]
        na_cnt = sum(s.isna())
        if na_cnt > 0:
            test_features[k] = s.fillna(0.0)

    predict_test_input_fn = lambda: my_input_fn_pred(test_features,
                                                     num_epochs=1,
                                                     shuffle=False)

    test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
    df_submit = pd.DataFrame()
    df_submit['daysOnMarket'] = np.array([item['predictions'][0] for item in test_predictions])
    list_df = list(df_submit['daysOnMarket'])
    # list_df =np.expm1(list_df)
    print(mean_absolute_error(df_test_label, list_df))


predict_test(DNN_model)

# 相关性分析：
# corrmat = data.corr()
# f, ax = plt.subplots(figsize=(100, 100))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()
#
# print(corrmat)
# # plt.figure(figsize=(20,10))
# # corrmat = data.corr()
# # k = 40 #number of variables for heatmap
# # cols = corrmat.nlargest(k, 'daysOnMarket')['daysOnMarket'].index
# # cm = np.corrcoef(data[cols].values.T)
# # sns.set(font_scale=1)
# # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# # plt.show()


