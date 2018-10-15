# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: iforest_to_detect_excetions.py
@time: 2018/10/15
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
from auto_ml import Predictor
from sklearn.metrics import mean_absolute_error


def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        # "city",
        "province",
        "price",
        "tradeTypeId",
        # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType'
    ]]
    # 是否只考虑tradeTypeId为1的数据
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    data = data.dropna(axis=0)
    data = data.reset_index(drop=True)

    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if isinstance(bedrooms,float):
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

def get_factorize_data(data):
    ## 对类别数据进行编码：
    for column in data.columns:
        if data[column].dtypes == 'object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            # data[column] = data[column].astype('str')
    return data


def use_iforest_to_get_normal_data(data):
    clf = IsolationForest(
        # max_samples=100, contamination='auto'
    )
    clf.fit(data)
    prediction_result = clf.predict(data)
    df_prediction = pd.DataFrame(prediction_result)
    print(df_prediction.head())
    inline_list = df_prediction[df_prediction[0] == 1].index
    return inline_list







if __name__ == '__main__':


    df_train = pd.read_csv('./input/month_567_data.csv')

    df_train = preprocess_data(df_train)
    print(df_train.shape)

    df_test_middle = pd.read_csv('./input/hose_info_201808_predict_2.csv')

    df_test_middle = preprocess_data(df_test_middle)
    print(df_test_middle.shape)
    # df_test_middle = date_processing(df_test_middle)
    origin_data = df_test_middle.reset_index()
    df_test_middle.to_csv('./origin_data_auto_ml.csv',index=False)


    df_train_factorize = get_factorize_data(df_train)
    # 去掉daysOnMarket 之后再去除异常值还是特别高；
    df_train_factorize =df_train_factorize.drop(columns='daysOnMarket')

    inline_list = use_iforest_to_get_normal_data(df_train_factorize)

    print(df_train.shape)
    df_inline_train = df_train[df_train.index.isin(inline_list)]
    print(df_inline_train.shape)



    df_test = df_test_middle.drop(columns='daysOnMarket')
    df_test_label = df_test_middle['daysOnMarket']

    value_list = []
    for i in range(len(df_train.columns)):
        value_list.append('categorical')

    column_description1 = {key: value for key in df_train.columns for value in value_list if
                           df_train[key].dtype == 'object'}
    column_description2 = {
        'daysOnMarket': 'output',
        'buildingTypeId': 'categorical',
        "tradeTypeId": 'categorical',
        # 'year': 'categorical',
        # 'month': 'categorical',

    }

    print(column_description1)
    # 合并两个字典
    column_descriptions = dict(column_description1, **column_description2)

    ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_inline_train, model_names='XGBRegressor')

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)

    # log还原
    # x = np.expm1(x)

    x_dataframe = pd.DataFrame(x,columns=['predictions'])
    merge_data = pd.concat((origin_data,x_dataframe),axis=1)
    merge_data_df = pd.DataFrame(merge_data)
    merge_data_df.to_csv('merge_data_auto_ml.csv',index=False)
    print(x_dataframe.describe())
    print(df_test_label.describe())

    print(mean_absolute_error(df_test_label,x))












