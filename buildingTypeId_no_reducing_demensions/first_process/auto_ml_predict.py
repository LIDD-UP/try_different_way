# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: auto_ml_predict.py
@time: 2018/8/30
"""


from auto_ml import Predictor
import  pandas as pd
from sklearn.model_selection import  train_test_split
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns

if __name__ == '__main__':

    df_train = pd.read_csv('./month_6_train_1.csv')
    # df_train['longitude'] = abs(df_train['longitude'] )
    df_train = df_train[[
        'longitude',
        # 'latitude',
        # 'buildingTypeId',
        'daysOnMarket',
        # 'bedrooms',
        # 'price'
    ]]

    # corrmat = df_train.corr()
    # plt.subplots(figsize=(12,9))
    # sns.heatmap(corrmat, vmax=0.9, square=True,annot=True)
    # plt.show()
    '''
    按照图中所展示的相关性热度图情况显示，只有latitude和，buildingTypeId ：corr的关系是正的
    其余的关系都是负的，尝试就用这两个特征预测出来的情况怎么样；
    '''

    df_test_middle = pd.read_csv('./test_data_1.csv')
    df_test_middle = df_test_middle[[
        'longitude',
        # 'latitude',
        # 'buildingTypeId',
        'daysOnMarket',
        # 'bedrooms',
        # 'price',
    ]]

    df_train =df_train.dropna()
    df_test_middle = df_test_middle.dropna()

    df_test = df_test_middle.drop(columns='daysOnMarket')
    df_test_label = df_test_middle['daysOnMarket']

    value_list = []
    for i in range(len(df_train.columns)):
        value_list.append('categorical')


    column_description1 = {key:value for key in df_train.columns for value in value_list if df_train[key].dtype =='object'}
    column_description2 = {
        'daysOnMarket': 'output',
        'buildingTypeId': 'categorical',

    }

    print(column_description1)
    column_descriptions = dict(column_description1, **column_description2)


    ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train,model_names='XGBRegressor')

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)
    print(mean_absolute_error(df_test_label,x))