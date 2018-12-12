#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: fillna.py
@time: 2018/7/12
"""

# 查看object类型数据的平均值是什么；

#去除缺失值后取均值和直接用mean结果是不是一样的；

import pandas as pd
import os
import numpy as np

# 获取文件
current_path = os.getcwd()
fil_name = '/dataset/{}.csv'.format('realtor_data_first1')
file_path = current_path + fil_name
data = pd.read_csv(file_path)
# 定义新文件存储位置
file_name_new = '/dataset/{}.csv'.format('realtor_data_fillna')
print(data['waitProcessContactDate'])

def column_null_len(dataframe,column):
    column_is_null = pd.isnull(dataframe[column])
    column_is_null_true = column_is_null[column_is_null]
    coulumn_is_null_len = len(column_is_null_true)
    return coulumn_is_null_len
print(column_null_len(data,'waitProcessContactDate')/len(data['waitProcessContactDate']))


# 判断isnull返回的dataframe结构，不能用bool型的去判断是否在dataframe中；
# print('True' in pd.isnull(data))
# x_test = pd.DataFrame([['a','b','c','d']],columns=['a','b','c','d'])
# x_test = np.nan
# print('a' in x_test)
# print(x_test)
# is_null = pd.isnull(x_test)
# print('is_null_dtypes',is_null.dtypes)
# print(pd.isnull(x_test))
# print(type(pd.isnull(x_test)))
# # print(False in pd.isnull(x_test['a']))
# true_null = True
# print(true_null not in pd.isnull(x_test['a']))
# bool_test = pd.DataFrame([[True,True,True,True]],columns=['a','b','c','d'])
# print(bool_test)
# print((True in bool_test))





# 判断mean是不是去掉缺失值求的（是）
# 对于object数据能不能用mean求均值；
# print(data['price'].mean())
# print(len(data['price']))
# price_new = data['price'].dropna()
# print(len(price_new))
# print(price_new.mean())


# 判断pandas 能否对非数值数据求众数；
# mode_test = pd.DataFrame(['a','b','a','c','c','c','d','a','a'])
# print(mode_test.mode())

#获取某一列的众数；这种方法太low了，有自带的方法；
# print(data['province'].mode())
# province_set = set()
# for i in data['province']:
#     province_set.add(i)
# print()

# 查看数据的类型有哪些
# data_dtypes = data.dtypes
# print(data_dtypes)
# 查看众数
# print(data['province'].mode())
# print()

# province_mode = data['province'].mode()
# province_mode_str = str(province_mode.values[0])
# print(province_mode_str)

# price = data['daysOnMarket']
# price_mean = data['price'].mean()
# price = data['price'].fillna(price_mean)
# print(price)





# province = data['province'].fillna(data['province'].mode())
# print(province)




# new_data = pd.DataFrame()
# # 保留原始文件
# count = 0
#
# for column in data.columns:
#     # 判定该列是否缺失
#     column_series = data[column]
#     # 找到缺失值的长度，用True无法做判断；
#     column_is_null = pd.isnull(column_series)
#     column_is_null_true = column_is_null[column_is_null]
#     column_is_null_len = len(column_is_null_true)
#
#     if column_is_null_len > 0:
#         # 对于object数据的填充
#         if column_series.dtype == 'object':
#             # 获取众数
#             column_mode = column_series.mode()
#             column_mode_str = column_mode.values[0]
#             print(type(column_mode))
#             # print(type(column_mode_str))
#             new_data[column] = column_series.fillna(column_mode_str)
#
#          # 对数值型数据的处理
#         if column_series.dtype == 'int64' or column_series.dtype == 'float64':
#             # 获取均值
#             column_mean = column_series.mean()
#             new_data[column] = column_series.fillna(column_mean)
#
#     if column_is_null_len == 0:
#         new_data[column]= column_series
#
#
#
# print(new_data)

# new_data.to_csv(file_name_new,index=False)






'''
#这里有一个错误的逻辑， 这是对series的一种处理，不是对DataFrame的处理，所以不需要考虑保留原始文件，
# 只需要定义一个DataFame 像字典一样向里面添加数字；
for column in data.columns:
    # 判定该列是否缺失
    column_series = data[column]
    # 找到缺失值的长度，用True无法做判断；
    column_is_null = pd.isnull(column_series)
    column_is_null_true = column_is_null[column_is_null]
    column_is_null_len = len(column_is_null_true)

    if column_is_null_len > 0:
        # 对于object数据的填充
        if column_series.dtype == 'object':
            count += 1
            # 获取众数
            column_mode = column_series.mode()
            这里有一个错误的逻辑，
            if count == 1:
                new_data[column] = column_series.fillna(column_mode)
            if count > 1:
                new_data[column].fillna(column_mode, inplace=True)
         # 对数值型数据的处理
        if column_series.dtype == 'int64' or column_series.dtype == 'float64':
            count += 1
            # 获取均值
            column_mean = column_series.mean()
            if count == 1:
                new_data[column] = column_series.fillna(column_mean)
            if count > 1:
                new_data[column].fillna(column_mean, inplace=True)


print(new_data)
'''



