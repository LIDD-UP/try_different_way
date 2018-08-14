# -*- coding:utf-8 _*-
""" 
@author:Administrator
@file: processing_data_function.py
@time: 2018/7/13
"""
'''
一些数据处理的函数，包括：
1：获取缺失值长度，
2：获取特征列不为空的长度
3：获取不为空数据大于多大长度的列
4：获取不为空数据大于多大长度的行 # 用到的可能性不大，感觉按照比率来更合适一些
5：按缺失值比率进行删除列
6：按缺失值比率进行删除行
7：对缺失值的填充
8：分组获取数据：主要针对缺失比率大，但是数据量大于10000的列，
'''
import pandas as pd
import os


# 数据的读取
def read_csv_data(filename):
    # 这种方式的文件读取方式仅限于py文件和dataset 文件属于同一级目录下
    current_path = os.getcwd() #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    data = pd.read_csv(file_path)
    return data


def to_csv_file(dataframe,filename):
    # 把dataframe转换成csv文件并存储到dataset下,仅限于py文件和dataset文件属于同一级目录下
    current_path = os.getcwd()  #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    dataframe.to_csv(file_path, index=False)



# 获取特征列为空的长度
def column_null_len(dataframe,column):
    column_is_null = pd.isnull(dataframe[column])
    column_is_null_true = column_is_null[column_is_null]
    coulumn_is_null_len = len(column_is_null_true)
    return coulumn_is_null_len


# 获取特征不为空的长度
def column_not_null_count(dataframe,column):
    column_is_null = pd.isnull(dataframe[column])
    column_is_null_true = column_is_null[column_is_null]
    column_is_null_len = len(column_is_null_true)
    column_not_null_len= len(dataframe[column])-column_is_null_len
    return column_not_null_len


# 获取不为空数据大于多大长度的列
def get_column_based_on_nan_len(dataframe,length):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        if len(dataframe[column]) != 0:
            column_is_null = pd.isnull(dataframe[column])
            column_is_null_true = column_is_null[column_is_null]
            column_is_null_len = len(column_is_null_true)
            column_not_null_len = len(dataframe[column])-column_is_null_len
            if column_not_null_len > length:
                new_data[column] = dataframe[column]
    return new_data


# 获取大于某个比率的列
def get_column_based_on_ration(dataframe,ratio):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        if len(dataframe[column]) != 0:
            column_is_null = pd.isnull(dataframe[column])
            column_is_null_true = column_is_null[column_is_null]
            column_is_null_len = len(column_is_null_true)
            if column_is_null_len / len(dataframe[column]) < ratio:
                new_data[column] = ratio[column]
    return new_data


# 删除缺失值大于某个比率的行
def delete_row_based_on_ration(dataframe,ratio):
    new_data = pd.DataFrame()
    count = 0  # 第几次删除行，第一次就要用原始数据进行赋值，后面就不用了，直接inplace=True，为了保留原始文件；
    for i in range(len(dataframe)):
        row_i_len = len(dataframe.loc[i])
        if row_i_len != 0:
            row_i = dataframe.loc[i]
            # 获取每行缺失值个数
            row_i_is_null = pd.isnull(row_i)
            row_i_is_null_true = row_i_is_null[row_i_is_null]
            row_i_is_null_len = len(row_i_is_null_true)
            # 若行的缺失值大于等于50% 就删除掉,所谓的删除就是获取满足条件的再用另外的dataframe进行存储
            if row_i_is_null_len / row_i_len >= 0.5:
                count += 1
                if count == 1:
                    new_data = dataframe.drop(index=i)
                    print('new_data_len_i', len(new_data))
                if count > 1:
                    new_data.drop(index=i, inplace=True)
    return new_data


# 对缺失值进行填充主要考虑到非数值数据和数值数据，没有对特殊类型如日期类型进行处理
def fill_na(dataframe):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        # 判定该列是否缺失
        column_series = dataframe[column]
        # 找到缺失值的长度，用True无法做判断；
        column_is_null = pd.isnull(column_series)
        column_is_null_true = column_is_null[column_is_null]
        column_is_null_len = len(column_is_null_true)

        if column_is_null_len > 0:
            # 对于object数据的填充
            if column_series.dtype == 'object':
                # 获取众数
                column_mode = column_series.mode()
                # 由于这种方式获取的是一个series ，需要将里面的值取出来；不然不会全部填充
                column_mode_str = column_mode.values[0]
                print(type(column_mode))
                new_data[column] = column_series.fillna(column_mode_str)
             # 对数值型数据均值填充
            if column_series.dtype == 'int64' or column_series.dtype == 'float64':
                # 获取均值
                column_mean = column_series.mean()
                new_data[column] = column_series.fillna(column_mean)
        if column_is_null_len == 0:
            new_data[column]= column_series
    return new_data


# 分组获取数据：主要针对缺失比率大，但是数据量大于10000的列，
