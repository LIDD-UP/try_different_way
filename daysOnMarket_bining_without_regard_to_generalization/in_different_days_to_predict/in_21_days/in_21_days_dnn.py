
# -*- coding:utf-8 _*-  
""" 
.
.
.
"""
import tensorflow as tf
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm, skew
import math
from sklearn import metrics


# 日志
tf.logging.set_verbosity(tf.logging.INFO)
data = pd.read_csv('./in_21_days.csv')
data['buildingTypeId'] = data['buildingTypeId'].astype('str')
data['tradeTypeId'] = data['tradeTypeId'].astype('str')

data = data.dropna()


_x = data.drop(columns=['daysOnMarket'])
_y = pd.DataFrame(data['daysOnMarket'])

# print(_x.head())



def split_data_with_ratio(data,train_ratio):
    len_data = int(len(data)*train_ratio)
    print(len_data)
    data_more = data.iloc[0:len_data,:]
    print(data_more.shape)
    data_less = data.iloc[len_data:,:]
    print(data_less.shape)
    return data_more,data_less

example ,example_predict = split_data_with_ratio(_x,0.9)
label,label_predict = split_data_with_ratio(_y,0.9)






# 生成训练和测试数据
def generate_input_data_dict(data):
    input_dict = {}
    for i in data.columns:
        if i =='bedrooms':
            input_dict[i] = data[i]
        else:
            input_dict[i] = data[i]
    return input_dict



# 定义连续型变量,直接用循环定义；
# 这里不考虑对连续变量的分桶操作，数据在生成的时候一定要明确类型，
# 对于类别型变量，先判断他的类别数再决定用不用hash，
# 在此之前还需继续查看官方文档的定义方式；
def generate_columns(data):
    # 拿到数据集里面的numeric变量
    numeric_columns = []
    class_columns =[]
    for column in data.columns:
        if data[column].dtype != 'object':
            numeric_column_real = tf.feature_column.numeric_column(column)
            numeric_columns.append(numeric_column_real)
        else:
            class_category_set = set(data[column])
            len_class_category_set = len(class_category_set)
            if len_class_category_set<100:
                class_column_real = tf.feature_column.categorical_column_with_vocabulary_list(column,[str(x) for x in class_category_set])

                class_columns.append(tf.feature_column.indicator_column(class_column_real))
            else:
                class_column_real = tf.feature_column.categorical_column_with_hash_bucket(column,100)
                class_columns.append(tf.feature_column.embedding_column(class_column_real,int(len_class_category_set**0.25)))
    columns =numeric_columns +class_columns

    return columns






# 定义模型（估计器）
estimator_model = tf.estimator.DNNRegressor(
    model_dir='D:/DNN_in21days',
    feature_columns=generate_columns(data.drop(columns='daysOnMarket')),
    hidden_units=[32,64,128,256,512,1024,2048],
    # hidden_units=[32,64],
)

# 获取预测输入：
def get_predict_input(example_predict):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            k: np.array(v) for (k, v) in generate_input_data_dict(example_predict).items()

        },
        y=np.array(label_predict),
        num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
        shuffle=True,
        batch_size=len(example_predict)
    )
    return predict_input_fn




# 为了保证交叉验证的效果，需要将训练和测试写在循环里面：
def get_input_to_train_and_test(example,label,estimator_model,batch_size,train_num):
    for period in range(train_num):
        example_train, example_test = split_data_with_ratio(example, 0.9)
        label_train, label_test = split_data_with_ratio(label, 0.9)
        # print(label_train.head())

        example_train_dict = generate_input_data_dict(example_train)
        print(example_train_dict.keys())

        def train_input_fn(example_train,is_shuffle=True,batch_size=batch_size):
            return tf.estimator.inputs.numpy_input_fn(
                x={
                    k: np.array(v) for (k,v) in generate_input_data_dict(example_train).items()
                },
                y=np.array(label_train),
                num_epochs=1,
                shuffle=is_shuffle,
                batch_size=batch_size,
            )


        def test_input_fn(example_test,is_shuffle=False,batch_size=batch_size):
            return tf.estimator.inputs.numpy_input_fn(
                x={
                    k: np.array(v) for (k, v) in generate_input_data_dict(example_test).items()
                },
                y=np.array(label_test),
                num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
                shuffle=is_shuffle,
                batch_size=batch_size,
                )

        print("Training model...")
        print("RMSE, RMSLE (on training data):")
        training_rmse = []
        validation_rmse = []
        training_rmsle = []
        validation_rmsle = []
        train_mean_absolute_error_list = []
        validataion_mean_absolute_error_list = []

        # 这里应该是是用的example_train的长度
        # 每一个num_epoch的训练步数
        steps_trains = int(len(example_train)/batch_size)+1
        estimator_model.train(input_fn=train_input_fn(example_train=example_train), steps=steps_trains)

        # 用原始数据进行预测
        training_predictions = estimator_model.predict(input_fn=train_input_fn(example_train=example_train,is_shuffle=False,batch_size=len(example_train)))
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        print('len-train-prediction', len(training_predictions))
        # Compute training loss RMSE.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(label_train, training_predictions))

        training_absolute_error = mean_absolute_error(label_train, training_predictions)

        # Compute training loss RMSLE.
        training_root_mean_squared_log_error = 1
            # math.sqrt(
            # metrics.mean_squared_log_error(label_train, training_predictions))

        # 当把数据集切分的时候，就做验证；

        validation_predictions = estimator_model.predict(input_fn=test_input_fn(example_test,is_shuffle=False,batch_size=len(example_test)))
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        print('len-validation-prediction', len(validation_predictions))
        # Compute validation loss RMSE.
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(label_test, validation_predictions))

        validataion_absolute_error = mean_absolute_error(label_test, validation_predictions)

        # Compute validation loss RMSLE.
        validation_root_mean_squared_log_error =1
        #  math.sqrt(
        #     metrics.mean_squared_log_error(label_test, validation_predictions))

        # Occasionally print the current loss.
        print(" train- period %02d : %0.2f, %0.4f,%0.4f" % (
            period, training_root_mean_squared_error,  training_absolute_error,training_root_mean_squared_log_error))

        print(" validation period %02d : %0.2f, %0.4f,%0.4f" % (
            period, validation_root_mean_squared_error, validataion_absolute_error,
            validation_root_mean_squared_log_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        training_rmsle.append(training_root_mean_squared_log_error)
        train_mean_absolute_error_list.append(training_absolute_error)
        validataion_mean_absolute_error_list.append(validataion_absolute_error)

        validation_rmse.append(validation_root_mean_squared_error)
        validation_rmsle.append(validation_root_mean_squared_log_error)

    print('len-train',len(label_train))
    print('len-test',len(label_test))
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.figure(figsize=(20, 5))
    # RMSE
    plt.subplot(1, 4, 4)
    plt.ylabel('mean_absolute_error')
    plt.xlabel('Periods')
    plt.title('mean_absolute_error')
    plt.plot(train_mean_absolute_error_list, label='train')

    plt.plot(validataion_mean_absolute_error_list, label="validation")
    plt.legend()

    plt.subplot(1, 4, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(training_rmse, label="training")

    plt.plot(validation_rmse, label="validation")
    plt.legend()
    # RMSLE
    plt.subplot(1, 4, 2)
    plt.ylabel("RMSLE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Logarithmic Error vs. Periods")
    plt.plot(training_rmsle, label="training")

    plt.plot(validation_rmsle, label="validation")
    plt.legend()
    # Target / Prediction
    plt.subplot(1, 4, 3)
    plt.ylabel("Target")
    plt.xlabel("Prediction")
    plt.title("Target vs. Prediction")
    lim = max(label_train.values)[0]

    # lim = max(lim, max(label_test.values))
    lim *= 1.05
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], alpha=0.5, color='red')
    plt.scatter(training_predictions, label_train, alpha=0.5, label="training")

    plt.scatter(validation_predictions, label_test, alpha=0.5, label="validation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return estimator_model


estimator_model = get_input_to_train_and_test(example,label,estimator_model,1,1000000)


# 预测
predict_iter = estimator_model.predict(input_fn=get_predict_input(example_predict))
# 循环次数根据测试数据数决定

list_value = []
for i in range(len(label_predict)):
    # 目前不知道这个dict_values([array([51.575745], dtype=float32)]) 数据怎么把值弄出来，就没有算精确度了
    x =float(list(predict_iter.__next__().values())[0])
    print(i, x)
    list_value.append(x)

print(list_value)
list_value = np.array(list_value)

list_value_series = pd.Series(list_value)
print(list_value_series.describe())

print('prediction_mean',np.mean(list_value))
print('label_mean',np.mean(label_predict))

print(mean_absolute_error(label_predict,list_value))


# 
