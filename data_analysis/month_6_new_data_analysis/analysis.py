# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: analysis.py
@time: 2018/8/23
"""

import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./month6_new.csv')
print(data.shape)
print(data.head())

# def label_encode(data):
#     for column in data.columns:
#         if data[column].dtypes=='object':
#             data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
#             data[column] = data[column].astype('str')
#     return data
#
# data = label_encode(data)
#
# # person相关性，corr()
# # correlation matrix
# corrmat = data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()





# # 先去除掉之前观测出来的无用特征：
# data = data.drop(columns=['id',
#                           'listingDate',
#                           'basement1','basement2',
#
#                           # 'heatType',
#                           'streetDirection',
#                            # 'room5','room5Length','room5Width',
#                            #  'room6','room6Length','room6Width',
#                             'room7','room7Length','room7Width',
#                             'room8','room8Length','room8Width',
#                             'room9','room9Length','room9Width',
#
#                           'cacIncluded','elevator','furnished','farmAgriculture',
#                           'lotSizeCode', 'waterIncluded','parkingIncluded','cableTVIncluded',
#
#                           ])
# # 处理bedrooms
# def process_bedrooms(data):
#     list_month = list(data['bedrooms'].astype('str'))
#     list_month_process = []
#     for i in list_month:
#         if i != 'nan':
#             list_month_process.append(eval(i))
#         else:
#             list_month_process.append(i)
#     data['bedrooms'] = pd.Series(list_month_process)
#     data['bedrooms'] = data['bedrooms'].astype('float')
#     return data
#
#
# data = process_bedrooms(data)
#
# # 分开tradetype
# data = data[data.tradeTypeId == 1]
#
#
# # 观测缺失值
# # msno.bar(data)
# # plt.show()
#
# # 检测不同数据量的保留，特征的保留情况
# def test_columns_num_with_diff_data(data,data_num):
#     columns_set = []
#     for column in data.columns:
#         if len(data[pd.notna(data[column]) ])>data_num:
#             columns_set.append(column)
#     data = data[columns_set]
#     return data,len(columns_set)
# data,len_columns = test_columns_num_with_diff_data(data,10000)
#
#
# def check_dropna_before_after(data):
#     print('before',data.shape)
#     print(data.head())
#     data = data.dropna()
#     print('after',data.shape)
#     print(data.head())
#     return data
#
#
#
# # sns.pairplot(data)
# # plt.show()
#
# # 现在一个特征一个特征的网上添加：在此之前要先确定特征的处理方式
# # 需不需要进行归一化或者是标准化操作；
#
#
# # 第一步：拿以前的来测试，[['price','buildingTypeId']]
# # data = data[['province','city','address','postalCode','longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
# data['buildingTypeId'] = data['buildingTypeId'].astype(str)
# data['bedrooms'] =data['bedrooms'].astype(str)
#
# print(data.shape)
# print(data.dtypes)
# def same_processing_way(data):
#     data = data[data.longitude < -10]
#     data = data[data.longitude > -140]
#     #
#     data = data[data.latitude > 43]
#     # data = data[data.tradeTypeId == 1]
#     #
#     data = data[data.price > 50000]
#     data = data[data.price < 2000000]
#     #
#     # data = data[data.buildingTypeId.isin([1, 3, 6])]
#     #
#     # list_bedrooms_new = []
#     # for i in data['bedrooms']:
#     #     if i > 6:
#     #         list_bedrooms_new.append(6)
#     #     else:
#     #         list_bedrooms_new.append(i)
#     # data['bedrooms'] = list_bedrooms_new
#     #
#     # data = data[data.daysOnMarket < 60]
#
#     data = data[data.drive!='']
#     # data = data[data.parkingIncluded != '']
#     return data
# data = same_processing_way(data)
# print(data.shape)
# data = check_dropna_before_after(data)
#
# print(data['heatType'].value_counts())
# print(data['drive'].value_counts())
#
#
# # for column in data.columns:
# #     if data[column].dtype=='object':
# #         print(data[column].value_counts())
#
#
#
# # 合并面积：
# def get_square(data=data,num_room_number=7):
#     rooms_colums_len =[]
#     rooms_colums_wid =[]
#     for i in range(1,num_room_number):
#         rooms_column_str_len = 'room' + str(i) + 'Length'
#         rooms_column_str_wid = 'room' + str(i) + 'Width'
#         rooms_colums_len.append(rooms_column_str_len)
#         rooms_colums_wid.append(rooms_column_str_wid)
#
#     for i,_len in enumerate(rooms_colums_len):
#         for j,wid in enumerate(rooms_colums_wid):
#             if i==j:
#                 name = 'rooms' + str(i + 1) + 'square'
#                 rooms_square_list = []
#                 for k in range(len(data)):
#                     rooms_square_list_k = list(data[_len])[k] * list(data[wid])[k]
#                     print(rooms_square_list_k)
#                     rooms_square_list.append(rooms_square_list_k)
#                 data[name] = rooms_square_list
#                 data = data.drop(columns=[_len,wid])
#     return data
# data = get_square(data,7)
# print(data.shape)
# print(data.head())
#
# print(data.dtypes)
# # data.to_csv('./dnn_data/second/second.csv',index=False)
#
#
#
#
#
#
#
#
# # corr_data = data.corr()
# # sns.heatmap(corr_data)
# # sns.pairplot(data)
# # msno.bar(data)
# # plt.show()
#
#
# # data.to_csv('./dnn_data/first.csv',index=False)
#
# # sns.pairplot(data)
# # plt.show()
