from auto_ml import Predictor
import  pandas as pd
from sklearn.model_selection import  train_test_split
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns


def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        "city",
        "province",
        "price",
        "tradeTypeId",
        "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType'
    ]]
    data = data[data.tradeTypeId == 1]
    data = data.dropna(axis=0)
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list
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

    df_train = pd.read_csv('./input/month_7_delisting_after_process_2.csv')
    df_train = preprocess_data(df_train)
    df_train = date_processing(df_train)



    df_test_middle = pd.read_csv('./input/month_8_delisting_data_after_process.csv')
    df_test_middle['ownerShipType'] = df_test_middle['ownershiptype']
    df_test_middle = df_test_middle.drop(columns='ownershiptype')
    df_test_middle = preprocess_data(df_test_middle)
    df_test_middle = date_processing(df_test_middle)
    origin_data = df_test_middle
    df_test_middle.to_csv('./origin_data_auto_ml.csv',index=False)



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
        'year': 'categorical',
        'month': 'categorical',

    }

    print(column_description1)
    column_descriptions = dict(column_description1, **column_description2)

    ml_predictor = Predictor(type_of_estimator='Regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train,model_names='XGBRegressor')

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)
    x_dataframe = pd.DataFrame(x,columns=['predictions'])
    merge_data = pd.concat((origin_data,x_dataframe),axis=1)
    merge_data_df = pd.DataFrame(merge_data)
    merge_data_df.to_csv('merge_data_auto_ml.csv',index=False)
    print(x_dataframe.describe())
    print(df_test_label.describe())

    print(mean_absolute_error(df_test_label,x))

