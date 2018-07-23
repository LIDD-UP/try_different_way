#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_one_hot_encoding.py
@time: 2018/7/23
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data= pd.read_csv('month_4_1.csv',header=0)
# column = data['column']
# column_list =set(column)
# print(column_list)

# column_dict = {}
# for index,value in enumerate(column_list):
#     column_dict[value] = index
# print(column_dict)
# 
# column_new = []
# for i in column:
#     column_new.append(column_dict[i])
# # print(len(column_new))
# # print(column_new)
# # print(column_new)
# column_new =np.array( pd.Series(column_new)).reshape(-1,1)
# enc = OneHotEncoder()
# column_enc = enc.fit_transform(column_new)
# column_env_array = column_enc.toarray()
# # print(column_enc.toarray().shape)
# column_dataframe = pd.DataFrame(column_env_array,columns=[col for col in column_dict.keys()])
# data_merge = pd.concat((data,column_dataframe),axis=1)
# print(data_merge)
# # print(len(column))
# # print(column)

def one_hot_encode_column(dataframe,column_name):
    # data = pd.read_csv('month_4_1.csv', header=0)
    # 获取需要进行one_hot 编码的列
    column = data[column_name]
    # 获取当前列的类别，直接用set就可以得到；
    column_set = set(column)
    print(len(column_set))
    # 将字符转化成标签并存入字典中
    column_dict = {}
    for index, value in enumerate(column_set):
        column_dict[value] = index
    # 开始转化成数字
    column_new = []
    for i in column:
        column_new.append(column_dict[i])
    column_new = np.array(pd.Series(column_new)).reshape(-1, 1)
    # 进行编码
    enc = OneHotEncoder()
    column_enc = enc.fit_transform(column_new)
    column_env_array = column_enc.toarray()
    # 合并成一个数据
    column_dataframe = pd.DataFrame(column_env_array, columns=[col for col in column_dict.keys()])
    data_merge = pd.concat((data, column_dataframe), axis=1)
    # 去掉转化过后的列：
    data_merge = data_merge.drop(columns=column_name)
    return data_merge


# new_data = one_hot_encode_column(data,'province')
# print(data.shape)
# print(new_data.shape)
# new_data1 = one_hot_encode_column(new_data,'city')
# print(new_data1.shape)
# print(new_data1)


# def process_all_data(dataframe):
#     columns = dataframe.columns
#     new_data = dataframe
#     for i in columns:
#         if new_data[i].dtype == 'object':
#             new_data_middle =one_hot_encode_column(new_data,i)
#             new_data = new_data_middle
#     return new_data

#
# new_data_final = process_all_data(data)
# print(new_data_final.shape)

# columns = data.columns
# new_data = data
# for i in columns:
#     if data[i].dtype == 'object':
#         new_data_middle = one_hot_encode_column(new_data,i)
#         new_data = new_data_middle
#
# print(new_data.shape)


# 对所有的字符列进行one_hot 编码：
