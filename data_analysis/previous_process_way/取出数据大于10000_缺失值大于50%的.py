#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: 取出数据大于10000_缺失值大于50%的.py
@time: 2018/7/13
"""
import pandas as pd
import os
import re



def read_csv_data(filename):
    # 这种方式的文件读取方式仅限于py文件和dataset 文件属于同一级目录下
    current_path = os.getcwd() #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    data = pd.read_csv(file_path)
    return data


def to_csv_file(dataframe,filename):
    # 把dataframe转换成csv文件并存储到dataset下,仅限于py文件和dataset文件属于同一级目录下s
    current_path = os.getcwd()  #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    dataframe.to_csv(file_path, index=False)




# 取出列有效数据大于10000 但是缺失比率大于%50，也就是说这时候需要找到缺失率大于%50 的列；s
def get_column_based_on_ration(dataframe,ratio):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        if len(dataframe[column]) != 0:
            column_is_null = pd.isnull(dataframe[column])
            column_is_null_true = column_is_null[column_is_null]
            column_is_null_len = len(column_is_null_true)
            if column_is_null_len / len(dataframe[column]) > ratio:
                new_data[column] = dataframe[column]
    return new_data

new_data = read_csv_data('more_than_10000')
new_data_processing = get_column_based_on_ration(new_data,0.5)
print(len(new_data_processing.columns))
to_csv_file(new_data_processing,'more_10000_and_more_middle')