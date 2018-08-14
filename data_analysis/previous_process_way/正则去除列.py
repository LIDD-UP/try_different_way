#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: 正则去除列.py
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




# 去除掉特征名包含某个单词或者是以什么开头，以什么结尾的列；
# 这里就去掉以Flag，mlsNumber  delislingDate postalCode processesAddress updateTimestamp结尾的；

data = read_csv_data('more_than_10000')
new_data = pd.DataFrame()
for column in data.columns:
    find_Flag = re.findall('Flag$',column)
    find_Number = re.findall('Number$',column)
    find_stamp = re.findall('stamp$',column)
    find_processes = re.findall('^processes',column)
    find_expected = re.findall('^expected',column)

    if len(find_Flag) == 0 and len(find_Number) == 0 and len(find_stamp) == 0 and len(find_processes) == 0 and len(find_expected) ==0:
        print(' not find ')
        new_data[column] = data[column]

print(len(data.columns))
print(len(new_data.columns))
to_csv_file(new_data,'re_delete')
