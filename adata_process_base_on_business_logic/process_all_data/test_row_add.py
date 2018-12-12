#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_row_add.py
@time: 2018/7/13
"""
import pandas as pd
import os
import random

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


data = read_csv_data('test_house_info')

list_add = ["province","city","address","postalCode","longitude","latitude","price","buildingTypeId","buildingTypeName","tradeTypeId","tradeTypeName","expectedDealPrice","listingDate","delislingDate","daysOnMarket"]

# new_data =pd.DataFrame()
# new_data.iloc[0] = data.i
# print(new_data)


# df = pd.DataFrame(columns=('lib', 'qty1', 'qty2'))
# print(df)
# for i in range(5):
#     df.loc[i] = [random.randint(-1, 1) for n in range(3)]
# print(df)

def delete_row_based_on_column_null(dataframe,column):
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


new_data = delete_row_based_on_column_null(data,'price')

print('datashape',data.shape)
print('new_data_shape',new_data.shape)