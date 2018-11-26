#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: wide_and_deep.py
@time: 2018/8/23
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


# 日志
tf.logging.set_verbosity(tf.logging.INFO)
data = pd.read_csv('./second.csv')
print(data.head())

data = data.dropna()
_x = data.drop(columns=['daysOnMarket'])
_y = data['daysOnMarket']

example, example_predict, label, label_predict = train_test_split(_x, _y, test_size=0.1)


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
    linear_columns=[]
    for column in data.columns:
        if data[column].dtype != 'object':
            numeric_column_real = tf.feature_column.numeric_column(column)
            numeric_columns.append(numeric_column_real)
        else:
            class_category_set = set(data[column])
            len_class_category_set = len(class_category_set)
            if len_class_category_set<100:
                class_column_real = tf.feature_column.categorical_column_with_vocabulary_list(column,class_category_set)
                linear_columns.append(class_column_real)
                class_columns.append(tf.feature_column.indicator_column(class_column_real))
            else:
                class_column_real = tf.feature_column.categorical_column_with_hash_bucket(column,100)
                class_columns.append(tf.feature_column.embedding_column(class_column_real,int(len_class_category_set**0.25)))
                linear_columns.append(class_column_real)
    columns =numeric_columns +class_columns
    return numeric_columns,class_columns,linear_columns






# 定义模型（估计器）
estimator_model = tf.estimator.DNNLinearCombinedRegressor(
    model_dir='./wide_and_deep',
    dnn_feature_columns=generate_columns(data.drop(columns='daysOnMarket'))[0],
    linear_feature_columns=generate_columns(data.drop(columns='daysOnMarket'))[0] +generate_columns(data.drop(columns='daysOnMarket'))[2],
    dnn_hidden_units=[32,64,128,256,512],
)

# 获取预测输入：
def get_predict_input(example_predict):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            k: np.array(v) for (k, v) in generate_input_data_dict(example_predict).items()

        },
        y=np.array(label_predict),
        num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
        shuffle=False,
        batch_size=len(example_predict)
    )
    return predict_input_fn




# 为了保证交叉验证的效果，需要将训练和测试写在循环里面：
def get_input_to_train_and_test(example,label,estimator_model,batch_size,train_num):
    for i in range(train_num):
        example_train, example_test, label_train, label_test = train_test_split(example, label, test_size=0.1)
        example_train_dict = generate_input_data_dict(example_train)
        print(example_train_dict)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
            k:np.array(v) for (k,v) in generate_input_data_dict(example_train).items()
            },
            y=np.array(label_train),
            num_epochs=1,
            shuffle=True,
            batch_size=batch_size,
        )

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
            k: np.array(v) for (k, v) in generate_input_data_dict(example_test).items()

            },
            y=np.array(label_test),
            num_epochs=1,  # 此处注意，如果设置成为None了会无限读下去；
            shuffle=False,
            batch_size=len(label_test),
        )

        # 训练
        steps_trains = int(len(example)/batch_size)+1
        estimator_model.train(input_fn=train_input_fn, steps=steps_trains)

        # 测试
        ev = estimator_model.evaluate(input_fn=test_input_fn)
        print('ev: {}'.format(ev))
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

# plt.plot(list_value,label='preds',c='red')
# plt.plot(label_test,label='true',c='blue')
# plt.show()
