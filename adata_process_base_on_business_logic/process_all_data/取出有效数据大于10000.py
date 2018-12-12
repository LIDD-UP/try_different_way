#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: 取出有效数据大于10000.py
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
    # 把dataframe转换成csv文件并存储到dataset下,仅限于py文件和dataset文件属于同一级目录下
    current_path = os.getcwd()  #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    dataframe.to_csv(file_path, index=False)





# # 取出列有效数据大于10000 的列
data = read_csv_data('realtor_data')
print(data)
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

#获取不为空数据量大于10000的数据
new_data = get_column_based_on_nan_len(data,10000)
print('new_data',len(new_data.columns))
# to_csv_file(new_data,'more_than_10000')

# ---------------------------------------------->>>>