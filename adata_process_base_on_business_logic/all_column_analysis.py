# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: all_column_analysis.py
@time: 2018/10/9
"""
'''
the py file is to process analysis all data
'''

import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# 观察在dropna 前后的形状变化
def before_after_dropna_shape(data):
    print(data.shape)
    data = data.dropna()
    print(data.shape)
    return data


test_data  = pd.read_csv('./input/month_8_d_more_columns.csv')

# 必须要去除的特征：
column_drop_list = [
    'id',
    'postalCode',
    'listingDate',
    ''
    'approxSquareFootage',
    'uffi',
    'waterfront',
    'farmAgriculture',
    'kitchens',
    'propertyType',
    'ammenitiesNearBy',
    'sizeInterior',
    'propertyTypeId',
    'kitchensPlus',
    'lotSizeCode',# 表示单位：feet，metres
    'ammenitiesNearBy',
    'propertyType',
    'sizeExterior',
    'ammenities',
    'sizeInterior',# 这个是单位不统一，但是可以通过lotSizeCode一起使用，要不全部转化成一样得刻度；
    'propertyTypeId',
]

column_save_list = [
    'province',
    # 'city', # 类别过多容易造成memory Error
    'longitude',
    'latitude',
    'price',
    'tradeTypeId',
    'buildingTypeId',
     "bedrooms",
    "bathroomTotal",
    # 'postalCode',
    'daysOnMarket',
    'ownerShipType',

]

deserve_save_but_num_less_list = [
    'furnished',
    'approxSquareFootage', # 是一个区间得数，如：1100-1299，那么这个数字改怎样取：最大，最小或者是平均值；
    'style',
    'community', # 类似于city类型得数据，类型有766个；
    'airConditioning', #
    'basement',# 地下室
    'frontingOn', # 面向得方向
    'familyRoom', #
    'drive',
    'farmAgriculture', # 太少了，只有几十个
    'fireplaceStove', # 2 w 左右
    'heatSource', # 数据量可以2w+
    'garageType', # 2 w+
    'pool',
    'parkingIncluded',
    'room1', # 表示得是房屋得种类(room1-room9)
    'sewers',# 这个数据比较少 1w+
    'uffi', # 特别得少：4k-
    'waterIncluded', # 1w+
    'approxAge',# 是一个范围数据：10-20这样的；1w-
    'laundryLevel',#1w+
    'propertyFeatures3',# 1w-
    'propertyFeatures4',# 1w-
    'propertyFeature5',# 1w-
    'propertyFeatures6',# 5k-
    'waterfront',# 1k-
    'zoningType',# 1w+
    'farmType',# 1w+
]





print(test_data.shape)
# print(test_data[pd.notna(test_data['waterfront'])])
# test_data = test_data[[column for column in test_data.columns if len(test_data[pd.notna(test_data[column])])>40000]]
#一个个网上加把：顺便查看最终的测试结果：


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
        'ownerShipType',

        'airConditioning',
    ]]

    return data


# for column in test_data.columns:
#     if test_data[column].dtype!='object':
#         print(test_data[column].value_counts())

# before_after_dropna_shape(test_data)




# msno.bar(test_data)
# plt.tight_layout()
# plt.show()

# 获取缺失值在1w以上得数据：
def get_data_more_1w(data):
    columns_list = []
    for column in data.columns:
        if len(data[pd.notna(data[column])])>20000:
            columns_list.append(column)
    return data[columns_list]


# new_data = get_data_more_1w(test_data)
# new_data = before_after_dropna_shape(new_data)
# print(new_data.head())
# print(get_data_more_1w(test_data).shape)


effective_feature = [
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
        'ownerShipType',

    # 'furnished', # 7953
    # 'approxSquareFootage', # 是一个区间得数，如：1100-1299，那么这个数字改怎样取：最大，最小或者是平均值； # 17209
    'style',
    # 'community', # 类似于city类型得数据，类型有766个；
    'airConditioning', #
    'washrooms', # 连续
    # 'bedroomsPlus', # 9633
    'basement1',# 地下室
    # 'basement2', # 这个drop掉之后只剩5k+了
    'frontingOn', # 面向得方向，drop掉之后有1w多:14270
    'familyRoom', #
    'drive', # 14270
    # 'farmAgriculture', # 太少了，只有几十个
    'fireplaceStove', # 2 w 左右
    'heatSource', # 数据量可以2w+
    'garageType', # 2 w+
    'kitchens',
    'parkingSpaces',
    'pool', # 这个偏少；14270
    'parkingIncluded',
    # 'room1', # 表示得是房屋得种类(room1-room9)
    'rooms',

    'sewers',# 这个数据比较少 1w+：14270
    # 'uffi', # 特别得少：4k-
    'waterIncluded', #
    'totalParkingSpaces',
    # 'approxAge',# 是一个范围数据：10-20这样的；1w-：11186
    # 'laundryLevel',#1w+：12710
    # 'propertyFeatures3',# 1w- ：10947
    # 'propertyFeatures4',# 1w-：9083
    # 'propertyFeature5',# 1w-
    # 'propertyFeatures6',# 5k-：4202
    # 'waterfront',# 1k-：1029
    # 'zoningType',# 1w+： # 这个不能用一drop掉之后就没了
    # 'farmType',# 1w+ ：# 这个不能用一drop掉之后就没了
]

# test_data = test_data[effective_feature]
# print(test_data.shape)
# before_after_dropna_shape(test_data)

# 获取交集得部分
# def get_intersection(data,sectiona,sectionb):
#     intersetion_list = list(set(sectiona).intersection(set(sectionb)))
#     return data[intersetion_list]

# print(get_intersection(new_data,new_data.columns,effective_feature).shape)


# test_data = test_data[effective_feature]
# before_after_dropna_shape(test_data)



# 24047这个级别得
# ：

effective_feature_2w = [
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
        'ownerShipType',
    'style',
    'community', # 类似于city类型得数据，类型有766个；
    'airConditioning', #
    'washrooms', # 连续
    'basement1',# 地下室
    'familyRoom', #
    'fireplaceStove', # 2 w 左右
    'heatSource', # 数据量可以2w+
    'garageType', # 2 w+
    'kitchens',
    'parkingSpaces',
    'parkingIncluded',
    'rooms',

    'waterIncluded', #
    'totalParkingSpaces',
]


effective_feature_1w = [
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
    'ownerShipType',

    'style',
    # 'community', # 类似于city类型得数据，类型有766个；
    'airConditioning',  #
    'washrooms',  # 连续
    'basement1',  # 地下室
    'familyRoom',  #
    'fireplaceStove',  # 2 w 左右
    'heatSource',  # 数据量可以2w+
    'garageType',  # 2 w+
    'kitchens',
    'parkingSpaces',
    'parkingIncluded',
    'rooms',
    'waterIncluded',  #
    'totalParkingSpaces',


    # 'frontingOn', # 面向得方向，drop掉之后有1w多:14270
    # 'drive', # 14270
    # 'pool', # 这个偏少；14270
    # 'sewers',# 这个数据比较少 1w+：14270


]


test_data_new = test_data[effective_feature_2w]
test_data_new = test_data_new.dropna()
print(test_data_new.shape)
test_data_inverse = test_data[~test_data.index.isin(test_data_new.index)]
print(test_data_inverse.shape)
print(test_data.shape)

test_data_inverse = test_data_inverse[[
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
    'ownerShipType',
]]
test_data_inverse = test_data_inverse.dropna()
print(test_data_inverse.shape)

test_data_new.to_csv('./input/more_column.csv',index=False)
test_data_inverse.to_csv('./input/less_column.csv',index=False)
