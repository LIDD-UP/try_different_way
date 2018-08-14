#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: processing_new_data.py
@time: 2018/7/12
"""
# 解决多了一列问题：加上index=False
# 还是为解决object类型数据转为csv文件后没有引号问题；

import pandas as pd
import os
import matplotlib.pyplot as plt
import missingno as msno

current_path = os.getcwd()
fil_name = '/dataset/{}.csv'.format('test_house_info')
file_path = current_path + fil_name
data = pd.read_csv(file_path)

print(data.dtypes)

data.to_csv('./realtor_data_first.csv',index=False,encoding='UTF-8')

current_path = os.getcwd()
fil_name = '/{}.csv'.format('new_test_house_info')
file_path = current_path + fil_name
data_new_house_info = pd.read_csv(file_path)


print(data_new_house_info.dtypes)

print(data.equals(data_new_house_info))