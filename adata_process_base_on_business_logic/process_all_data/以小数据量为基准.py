#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: 以小数据量为基准.py
@time: 2018/7/13
"""

import pandas as pd
import os
import re


'''
注意以drop的方式来删除行没有以赋值方式获取数据快；
直接用new_data.loc[i] = data.loc[i] 要快点；
'''


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




# 以缺失值在%50以上，有效数据大于10000中的列为基准在有效数据大于10000上获取有效数据：

# 获取列：

more_than_10000_data = read_csv_data('test_house_info')
# more_10000_and_more_middle_data = read_csv_data('more_10000_and_more_middle')
# based_column = more_10000_and_more_middle_data.columns
# print(based_column[0])
# contactFirstName

# 还是要拿more_than_10000 数据，以10000有效数据到，缺失率大于%50 的列（不为空的）为基准获取样本数据；




def delete_row_based_on_column_null_fast_way(dataframe,column):
    new_data = pd.DataFrame(columns=list_add)
    data_length = len(dataframe)
    for i in range(data_length):
            row_i = dataframe.loc[i]
            # 判断指定位置的数书否缺失
            column_row_i = row_i[column]
            if pd.notna(column_row_i):
                new_data.loc[i] = dataframe.loc[i]
                print('new_data_len_i', len(new_data))
    return new_data
# new_data = delete_row_based_on_column_null(data,'price')
# print('datashape',data.shape)
# print('new_data_shape',new_data.shape)





def delete_row_based_on_column_null(dataframe,column):
    new_data = pd.DataFrame()
    count = 0  # 第几次删除行，第一次就要用原始数据进行赋值，后面就不用了，直接inplace=True，为了保留原始文件；
    for i in range(len(dataframe)):
            row_i = dataframe.loc[i]
            # 判断指定位置的数书否缺失
            column_row_i = row_i[column]
            if pd.isna(column_row_i):
                count += 1
                if count == 1:
                    new_data = dataframe.drop(index=i)
                    print('new_data_len_i', len(new_data))
                if count > 1:
                    new_data.drop(index=i, inplace=True)
    return new_data


new_data = delete_row_based_on_column_null(more_than_10000_data,'price')
print(more_than_10000_data.shape)
print(new_data.shape)


# # 循环处理 获取每一次处理之后的文件；
# for column in more_10000_and_more_middle_data.columns:
#     new_data = delete_row_based_on_ration(more_than_10000_data,column)
#     print(column,':的形状',new_data.shape)
#     new_data.tocsv(new_data,'{}'.format(column))