# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: auto_ml_to_predict.py
@time: 2018/9/12
"""
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


def pre_process_features(canada_housing_data):
    canada_housing_data = canada_housing_data.dropna(axis=0)
    bedrooms_list = []
    for bedrooms in canada_housing_data["bedrooms"]:
        bedrooms_list.append(int(eval(bedrooms)))
    canada_housing_data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in canada_housing_data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    canada_housing_data["bathroomTotal"] = bathroom_total_list
    canada_housing_data = canada_housing_data[canada_housing_data['daysOnMarket'] <= 365]
    canada_housing_data = canada_housing_data[canada_housing_data['longitude'] != 0]
    canada_housing_data = canada_housing_data[canada_housing_data['latitude'] != 0]
    canada_housing_data = canada_housing_data[canada_housing_data['tradeTypeId'] == 1]
    canada_housing_data = canada_housing_data[canada_housing_data['longitude'] >= -145]
    canada_housing_data = canada_housing_data[canada_housing_data['longitude'] <= -45]
    canada_housing_data = canada_housing_data[canada_housing_data['latitude'] >= 40]
    canada_housing_data = canada_housing_data[canada_housing_data['latitude'] <= 90]
    canada_housing_data = canada_housing_data[canada_housing_data['price'] > 1]
    canada_housing_data = canada_housing_data.dropna(axis=0)

    selected_features = canada_housing_data[
        ["longitude",
         "latitude",
         # "city",
         # "province",
         "price",
         # "propertyType",
         "tradeTypeId",
         "listingDate",
         "buildingTypeId",
         "bedrooms",
         "bathroomTotal",
         'daysOnMarket',
         ]]

    # data_lenth = len(canada_housing_data)
    # processed_features = (selected_features.head(int(data_lenth*0.5))).copy()
    processed_features = selected_features.copy()
    # processed_features["longitude"] = round(processed_features["longitude"], 2)
    # processed_features["latitude"] = round(processed_features["latitude"], 2)
    # postCodeList = []
    # for item in canada_housing_data["postalCode"]:
    #     postCodeList.append(item.split(' ')[0])
    # processed_features["postalCodeThreeStr"] = postCodeList
        # list(set(postCodeList))
    listingDateMonth = []
    for item in canada_housing_data["listingDate"]:
        # print(item)
        if '-' in item:
            listingDateMonth.append(int(item.split('-')[1]))
        else:
            listingDateMonth.append(int(item.split('/')[1]))
    processed_features["listingDataMonth"] = listingDateMonth

    return processed_features




if __name__ == '__main__':
    df_train = pd.read_csv('./input/train.csv')
    df_train = pre_process_features(df_train)
    df_test_middle = pd.read_csv('./input/test.csv')
    df_test_middle = pre_process_features(df_test_middle)


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

    ml_predictor.train(df_train,model_names='DeepLearningRegressor')

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)
    print(mean_absolute_error(df_test_label,x))

