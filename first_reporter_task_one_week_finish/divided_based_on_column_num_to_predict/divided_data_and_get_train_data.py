# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/9/25
"""
import pandas as pd

pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data_origin = pd.read_csv("./input/month_567_d_more_columns.csv")
test_data_origin = pd.read_csv("./input/month_8_d_more_columns.csv")



# 预测数据的拆分：#
# 首先找到基本数据：通过下标：基本数据就是之前的；





# 预处理数据
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
        'postalCode',
        'daysOnMarket',
        'ownerShipType',


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


def preprocess_data_add_column(data):
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
        'postalCode',
        'daysOnMarket',
        'ownerShipType',

        'airConditioning',
        'storiesTotal',
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



if __name__ == '__main__':
    # train data 基本数据
    train_data_base_less = preprocess_data(train_data_origin)
    # 怎加一个特征的训练数据
    train_data_add = preprocess_data_add_column(train_data_origin)
    # test 基本数据：也就是总数据，每次增加一列都是在这个数据的基础上去除增加一列的下标
    test_data_base = preprocess_data(test_data_origin)
    # test_新数据
    test_data_new = preprocess_data_add_column(test_data_origin)
    # test data 去除新数据的那一部分
    test_data_base_del_new_less = test_data_base[~test_data_base.index.isin(test_data_new.index)]
    # 真正的新数据
    test_data_new_true_add = test_data_new[test_data_new.index.isin(test_data_base.index)]

    train_data_base_less.to_csv('./input/train_less.csv', index=False)
    test_data_base_del_new_less.to_csv('./input/test_less.csv', index=False)
    train_data_add.to_csv('./input/train_add.csv', index=False)
    test_data_new_true_add.to_csv('./input/test_add.csv', index=False)











