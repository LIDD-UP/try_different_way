# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data_and_use_auto_ml_to_predict_less.py
@time: 2018/9/21
"""
from auto_ml import Predictor
import  pandas as pd
from sklearn.model_selection import  train_test_split
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import numpy as np
from auto_ml import K


def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        # "city",
        # "province",
        "price",
        "tradeTypeId",
        # # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType',
        # 'projectDaysOnMarket',
        'district',

        # 以下就是用于测试得新得特征；
        'style', #22.769283885157083
        'community', # 类似于city类型得数据，类型有766个； #22.38147912725983
        'airConditioning', #22.755048806968883
        'washrooms', # 连续 #23.691205780782205
        'basement1',# 地下室22.797430800725444
        'familyRoom', # 22.794731300998404
        'fireplaceStove', # 2 w 左右 #22.82878318024665
        'heatSource', # 数据量可以2w+# 22.75554140962404
        'garageType', # 2 w+ #22.79707321027956
        'kitchens', # 22.79393809434976
        'parkingSpaces', #22.807931672409705
        #
        'parkingIncluded',#22.786586056260784
        'rooms',# 22.785397232054713

        'waterIncluded', # 22.80653144493355
        'totalParkingSpaces', # 22.81551411353129
        #
        # 'frontingOn',  # 面向得方向，drop掉之后有1w多:14270
        # 'drive',  # 14270
        # 'pool',  # 这个偏少；14270
        # 'sewers',  # 这个数据比较少 1w+：14270

        # more column
        # 'room3',

]]
    # 是否只考虑tradeTypeId为1的数据
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    print(data.shape)
    data = data.dropna(axis=0)
    print(data.shape)
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if isinstance(bedrooms,float):
            bedrooms_list.append(int(bedrooms))
        elif isinstance(bedrooms,int):
            bedrooms_list.append(int(bedrooms))
        else:
            bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list

    # 将price做log变换
    # data['price'] = np.log1p(data['price'])
    return data


def date_processing(data):
    list_date = list(data['listingDate'])
    year_list = []
    month_list = []
    day_list = []
    for date in list_date:
        # print(date)
        list_break = date.split('/')
        year_list.append(int(list_break[0]))
        month_list.append(list_break[1])
        day_list.append(list_break[2])
    data['year'] = year_list
    data['month'] = month_list
    # data['day'] = day_list
    data = data.drop(columns='listingDate')

    return data


def compute_ratio(data):
    data_10 = []
    data_20 = []
    data_30 = []
    data_more = []

    # data = data.drop(columns=['index'])

    # test_column = 'projectDaysOnMarket'
    test_column = 'predictions'

    for i in range(len(data)):
        # print(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 10:
            data_10.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 10 and abs(
                data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 20:
            data_20.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 20 and abs(
                data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 30:
            data_30.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 30:
            data_more.append(i)

    print(len(data_10) / len(data))
    print(len(data_20) / len(data))
    print(len(data_30) / len(data))
    print(len(data_more) / len(data))
    print(mean_absolute_error(data[test_column], data['daysOnMarket']))


def compute_ratio2(data):
    test_column = 'predictions'
    data_len = len(data)
    print(data[abs(data.predictions - data.daysOnMarket) <= 10]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 10 and abs(
                data.predictions - data.daysOnMarket) <= 20]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 20 and abs(
                data.predictions - data.daysOnMarket) <= 30]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 30]/data_len)
    print(mean_absolute_error(data[test_column], data['daysOnMarket']))






if __name__ == '__main__':
    df_train = pd.read_csv('./input/treb_toronto_3to8_1.csv')
    # df_train = pd.read_csv('./input/treb_toronto_678_1.csv')
    # df_train = pd.read_csv('./input/treb_toronto_78_1.csv')

    # df_train = pd.read_csv('./input/treb_all_column_month_3to8.csv')
    print(df_train.shape)
    df_train = preprocess_data(df_train)

    # 用于预测原始数据
    df_train_middle = df_train
    df_train_prediction = df_train_middle.drop(columns=['daysOnMarket'])
    # 用于保存原始的数据
    origin_data_train = df_train.reset_index(drop=True)


    print(df_train.shape)




    # df_test_middle = pd.read_csv('./input/treb_test_month_9.csv')
    # df_test_middle = pd.read_csv('./input/treb_test_all_column_month_9.csv')
    df_test_middle = pd.read_csv('./input/treb_toronto_9.csv')
    # df_test_middle = pd.read_csv('./input/treb_toronto_10.csv')
    from auto_ml import load_ml_model

    print(df_test_middle.shape)

    df_test_middle = preprocess_data(df_test_middle)
    print(df_test_middle.shape)
    origin_data = df_test_middle.reset_index(drop=True)




    df_train =df_train.dropna()
    # # 将daysOnMarket 做特征转换
    # df_train['daysOnMarket'] = np.log1p(df_train['daysOnMarket'])
    df_test_middle = df_test_middle.dropna()
    # df_test_middle = df_test_middle.drop(columns='projectDaysOnMarket')

    df_test = df_test_middle.drop(columns='daysOnMarket')

    df_test_label = df_test_middle['daysOnMarket']

    value_list = []
    for i in range(len(df_train.columns)):
        value_list.append('categorical')


    column_description1 = {key:value for key in df_train.columns for value in value_list if df_train[key].dtype =='object'}
    column_description2 = {
        'daysOnMarket': 'output',
        'buildingTypeId': 'categorical',
        "tradeTypeId": 'categorical',
        # 'bedrooms': 'categorical',
        # 'year': 'categorical',
        # 'month': 'categorical',

    }

    print(column_description1)
    # 合并两个字典
    column_descriptions = dict(column_description1, **column_description2)

    ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train,model_names='KerasRegressor') # KerasRegressor XGBRegressor
    ml_predictor.save('model_auto_ml.h5')


    # 预测预测数据
    x = ml_predictor.predict(df_test)
    x_dataframe = pd.DataFrame(x,columns=['predictions'])
    merge_data = pd.concat((origin_data,x_dataframe),axis=1)
    merge_data_df = pd.DataFrame(merge_data)
    merge_data_df.to_csv('./merge_data_bak/merge_data_auto_ml.csv',index=False)
    print(x_dataframe.describe())
    print(df_test_label.describe())

    print(mean_absolute_error(df_test_label,x))
    compute_ratio(merge_data_df)
    # compute_ratio2(merge_data_df)

    # 预测训练数据
    train_prediction = ml_predictor.predict(df_train_prediction)
    train_dataframe = pd.DataFrame(train_prediction,columns=['trainPrediction'])
    merge_train_data = pd.concat((origin_data_train,train_dataframe),axis=1)
    merge_train_data_df = pd.DataFrame(merge_train_data)
    merge_train_data_df.to_csv('./merge_data_bak/merge_train_data.csv',index=False)








