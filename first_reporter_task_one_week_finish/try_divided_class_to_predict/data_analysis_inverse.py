# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_quantile_based_buckets.py
@time: 2018/9/25
"""
import pandas as pd

pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv("./input/month_567_data.csv")
test_data = pd.read_csv("./input/hose_info_201808_predict_2.csv")




# 预处理数据
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
        'postalCode',
        'daysOnMarket',
        'ownerShipType'
    ]]
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    print('data shape=%s before dropna' % (str(data.shape)))
    data = data.dropna(axis=0)
    print('data shape=%s after dropna' % (str(data.shape)))
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if isinstance(bedrooms, float):
            bedrooms_list.append(int(bedrooms))
        else:
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
        if '/' in date:
            list_break = date.split('/')
            day_list.append(int(list_break[0]))
            month_list.append(int(list_break[1]))
            year_list.append(int(list_break[2]))
        elif '-' in date:
            list_break = date.split('-')
            year_list.append(int(list_break[0]))
            month_list.append(int(list_break[1]))
            day_list.append(int(list_break[2]))
    data['year'] = year_list
    data['month'] = month_list
    # data['day'] = day_list
    data = data.drop(columns='listingDate')

    return data


def show_value_counts(data, columns):
    for column in columns:
        print(data[column].value_counts())
    print(data.shape)


'''
就目前来讲，数据量最大的和最小的相差最多10倍情况下来取数据：
city：

'''


# 处理城市
def process_city(train_data, test_data, threshold_value):
    print('city nums before process:', len(set(train_data['city'])))
    city_list = set(train_data['city'])
    list_fill = []
    for city in city_list:
        if len(train_data[train_data.city == city]) > threshold_value:
            list_fill.append(city)
    print('city nums after process:', len(list_fill))
    # 只要满足条件的数据
    train_data = train_data[train_data.city.isin(list_fill)]
    print(train_data.shape)
    test_data = test_data[test_data.city.isin(list_fill)]
    print(test_data.shape)

    return train_data, test_data


# 处理postalCode
def get_category_class_bigger_than_threshold_value_postalcode(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('postalCode nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.postalCode == value]) > threshold_value:
            list_fill.append(value)
    print('postalCode nums after process:', len(list_fill))
    data = data[data.postalCode.isin(list_fill)]
    test_data = test_data[test_data.postalCode.isin(list_fill)]
    return data, test_data


# 处理省份
def get_category_class_bigger_than_threshold_value_province(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('province nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.province == value]) > threshold_value:
            list_fill.append(value)
    print('province nums after process:', len(list_fill))
    data = data[data.province.isin(list_fill)]
    test_data = test_data[test_data.province.isin(list_fill)]
    return data, test_data


# 处理buildingTypeId
def get_category_class_bigger_than_threshold_value_buildingTypeId(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('buildingTypeId nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.buildingTypeId == value]) > threshold_value:
            list_fill.append(value)
    print('buildingTypeId nums after process:', len(list_fill))
    data = data[data.buildingTypeId.isin(list_fill)]
    test_data = test_data[test_data.buildingTypeId.isin(list_fill)]
    return data, test_data


# 处理 ownerShipType
def get_category_class_bigger_than_threshold_value_ownerShipType(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('ownerShipType nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.ownerShipType == value]) > threshold_value:
            list_fill.append(value)
    print('ownerShipType nums after process:', len(list_fill))
    data = data[data.ownerShipType.isin(list_fill)]
    test_data = test_data[test_data.ownerShipType.isin(list_fill)]
    return data, test_data


# 处理 bedrooms
def get_category_class_bigger_than_threshold_value_bedrooms(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('bedrooms nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.bedrooms == value]) > threshold_value:
            list_fill.append(value)
    print('bedrooms nums after process:', len(list_fill))
    data = data[data.bedrooms.isin(list_fill)]
    test_data = test_data[test_data.bedrooms.isin(list_fill)]
    return data, test_data


# 处理 bathroomTotal
def get_category_class_bigger_than_threshold_value_bathroomTotal(data, test_data, column, threshold_value):
    column_list = set(data[column])
    print('bathroomTotal nums before process:', len(column_list))
    list_fill = []
    for value in column_list:
        if len(data[data.bathroomTotal == value]) > threshold_value:
            list_fill.append(value)
    print('bathroomTotal nums after process:', len(list_fill))
    data = data[data.bathroomTotal.isin(list_fill)]
    test_data = test_data[test_data.bathroomTotal.isin(list_fill)]
    return data, test_data


if __name__ == '__main__':
    # 预处理数据
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    train_data = date_processing(train_data)
    test_data = date_processing(test_data)

    # train_data = train_data[train_data.year == 2018]
    # test_data = test_data[test_data.year == 2018]

    print(train_data.head())
    print(test_data.head())
    train_data_origin= train_data
    test_data_origin= test_data


    category_variable = ['province', 'city',
                         'tradeTypeId', 'buildingTypeId',
                         'bedrooms', 'bathroomTotal',
                         # 'postalCode',
                         'ownerShipType',
                         'year', 'month',
                         # 'daysOnMarket'
                         ]

    # 处理city
    # print(train_data.shape)
    # train_data, test_data = process_city(train_data,test_data,10) # 50:366:shape:15000


    # # # # 处理省份
    # train_data,test_data = get_category_class_bigger_than_threshold_value_province(train_data,test_data,'province',1000)

    # train_data = pd.read_csv('./input/train.csv')
    # test_data = pd.read_csv('./input/test.csv')
    # print(train_data['province'].value_counts())


    # train_data, test_data = get_category_class_bigger_than_threshold_value_buildingTypeId(train_data, test_data,
    #                                                                                       'buildingTypeId', 10000)
    # # # # 处理ownerShipType
    # train_data, test_data = get_category_class_bigger_than_threshold_value_ownerShipType(train_data, test_data,
    #                                                                                      'ownerShipType', 1000)
    # # # # 处理bedrooms
    # train_data, test_data = get_category_class_bigger_than_threshold_value_bedrooms(train_data, test_data, 'bedrooms',
    #                                                                                 10000)
    # # # # 处理bathroomTotal
    # train_data, test_data = get_category_class_bigger_than_threshold_value_bathroomTotal(train_data, test_data,
    #                                                                                      'bathroomTotal', 10000)

    # # 处理postalCode
    # train_data, test_data = get_category_class_bigger_than_threshold_value_postalcode(train_data,test_data, 'postalCode', 10)
    # train_data = train_data[train_data.tradeTypeId==1]
    # test_data = test_data[test_data.tradeTypeId==1]

    # show_value_counts(train_data,category_variable)
    # train_data = train_data[train_data.year == 2018]
    # test_data = test_data[test_data.year == 2018]


    # 现在进行区分不同的省份进行预测：
    # train_data = train_data[train_data.province=='Ontario']
    # test_data = test_data[test_data.province=='Ontario']
    train_data = train_data[train_data.city=='Toronto']
    test_data = test_data[test_data.city=='Toronto']

    print(train_data.head())
    print(test_data.head())
    train_data_inverse = train_data_origin[~(train_data_origin.index.isin(train_data.index))]
    test_data_inverse = test_data_origin[~(test_data_origin.index.isin(test_data.index))]
    print(train_data_origin.shape)
    print(test_data_origin.shape)
    print(train_data.shape)
    print(test_data.shape)
    print(train_data_inverse.shape)
    print(test_data_inverse.shape)

    train_data.to_csv('./input/train.csv', index=False)
    test_data.to_csv('./input/test.csv', index=False)
    train_data_inverse.to_csv('./input/train_inverse.csv', index=False)
    test_data_inverse.to_csv('./input/test_inverse.csv', index=False)











