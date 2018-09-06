# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: auto_ml_to_predict.py
@time: 2018/9/5
"""
from auto_ml import Predictor
import  pandas as pd
from sklearn.model_selection import  train_test_split
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

'''
auto_ml 不知道有没有做交叉验证和调参，
如果没做该怎么办，需要看官方的文档进行确认；
'''


# 画图函数；
def plot_prediction_and_label(train_predict,train_label,test_predict,test_label):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.ylabel("Target")
    plt.xlabel("Prediction")
    plt.title("test-Target vs. Prediction")
    lim = max(test_label)
    lim *= 1.05
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], alpha=0.5, color='red')
    plt.scatter(test_predict, test_label, alpha=0.5, label="training")

    plt.legend()
    plt.tight_layout()

    # Target / Prediction
    plt.subplot(1, 2, 2)
    plt.ylabel("Target")
    plt.xlabel("Prediction")
    plt.title("train-Target vs. Prediction")
    lim = max(train_label)
    lim *= 1.05
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], alpha=0.5, color='red')
    plt.scatter(train_predict, train_label, alpha=0.5, label="training")

    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('/eval_bedroom_drop_id.csv')
    data = data.drop(columns=['province','city','address','postalCode'])
    print(data.shape)

    # 保留多少特征用于训练

    # data = data['']

    data = data.dropna()

    # 用多少样本进行训练
    # data = data.iloc[:,40:80]



    df_train, df_test_middle = train_test_split(data,test_size=0.1)
    df_train_example = df_train.drop(columns='daysOnMarket')
    df_train_label = df_train['daysOnMarket']
    df_test = df_test_middle.drop(columns='daysOnMarket')
    df_test_label = df_test_middle['daysOnMarket']

    value_list = []
    for i in range(len(data.columns)):
        value_list.append('categorical')


    column_description1 = {key:value for key in data.columns for value in value_list if data[key].dtype =='object'}
    column_description2 = {
        'daysOnMarket': 'output',
        'buildingTypeId': 'categorical'
    }

    print(column_description1)
    column_descriptions = dict(column_description1, **column_description2)


    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train)
    train_predict = ml_predictor.train(df_train_example)

    test_predict = ml_predictor.predict(df_test)
    print(mean_absolute_error(df_test_label,test_predict))

    # 画图分析拟合情况
    plot_prediction_and_label(train_predict,df_train_label,test_predict,df_test_label)

