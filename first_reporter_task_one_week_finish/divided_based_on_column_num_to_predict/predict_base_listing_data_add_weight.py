# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: predict_base_listing_data_add_weight.py
@time: 2018/10/9
"""

from auto_ml import Predictor
import  pandas as pd
from sklearn.model_selection import  train_test_split
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import numpy as np


def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        # "city",
        "province",
        "price",
        "tradeTypeId",
        # # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType',

        # 以下就是用于测试得新得特征；
        # 'style', #22.769283885157083
        # 'community', # 类似于city类型得数据，类型有766个； #22.38147912725983
        # 'airConditioning', #22.755048806968883
        # 'washrooms', # 连续 #23.691205780782205
        # 'basement1',# 地下室22.797430800725444
        # 'familyRoom', # 22.794731300998404
        # 'fireplaceStove', # 2 w 左右 #22.82878318024665
        # 'heatSource', # 数据量可以2w+# 22.75554140962404
        # 'garageType', # 2 w+ #22.79707321027956
        # 'kitchens', # 22.79393809434976
        # 'parkingSpaces', #22.807931672409705
        #
        # 'parkingIncluded',#22.786586056260784
        # 'rooms',# 22.785397232054713
        #
        # 'waterIncluded', # 22.80653144493355
        # 'totalParkingSpaces', # 22.81551411353129
        #
        # 'frontingOn',  # 面向得方向，drop掉之后有1w多:14270
        # 'drive',  # 14270
        # 'pool',  # 这个偏少；14270
        # 'sewers',  # 这个数据比较少 1w+：14270
]]
    # 是否只考虑tradeTypeId为1的数据
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    print(data.shape)
    data = data.dropna(axis=0)
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







if __name__ == '__main__':

    # df_train = pd.read_csv('./input/month_2to7_d_more_columns.csv')
    df_train = pd.read_csv('./input/listing_17.csv')
    print(df_train.shape)
    # df_train = pd.read_csv('./data_process/process_fliers.csv')
    # df_train['ownerShipType'] = df_train['ownershiptype']
    # df_train = df_train.drop(columns='ownershiptype')
    df_train = preprocess_data(df_train)
    print(df_train.shape)
    # df_train = date_processing(df_train)



    df_test_middle = pd.read_csv('./input/month_8_d_more_columns.csv')
    print(df_test_middle.shape)
    # df_test_middle['ownerShipType'] = df_test_middle['ownershiptype']
    # df_test_middle = df_test_middle.drop(columns='ownershiptype')
    df_test_middle = preprocess_data(df_test_middle)
    print(df_test_middle.shape)
    # df_test_middle = date_processing(df_test_middle)
    origin_data = df_test_middle.reset_index()
    # df_test_middle.to_csv('./origin_data_auto_ml.csv',index=False)



    df_train =df_train.dropna()
    # # 将daysOnMarket 做特征转换
    # df_train['daysOnMarket'] = np.log1p(df_train['daysOnMarket'])
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
        "tradeTypeId": 'categorical',
        # 'bedrooms': 'categorical',
        # 'year': 'categorical',
        # 'month': 'categorical',

    }

    print(column_description1)
    # 合并两个字典
    column_descriptions = dict(column_description1, **column_description2)

    ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train,model_names='XGBRegressor')

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)

    # log还原
    # x = np.expm1(x)

    x_dataframe = pd.DataFrame(x,columns=['predictions'])
    merge_data = pd.concat((origin_data,x_dataframe),axis=1)
    merge_data_df = pd.DataFrame(merge_data)
    merge_data_df.to_csv('./merge_data_bak/merge_data_auto_ml_test_listing_17.csv',index=False)
    print(x_dataframe.describe())
    print(df_test_label.describe())

    print(mean_absolute_error(df_test_label,x))





