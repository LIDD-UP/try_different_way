#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: delete_column_with_numerical_NaN.py
@time: 2018/7/11
"""



import pandas as pd
import os
import matplotlib.pyplot as plt
import missingno as msno

current_path = os.getcwd()
fil_name = '/dataset/{}.csv'.format('realtor_data')
file_path = current_path + fil_name
data = pd.read_csv(file_path)

# print(type(data.isnull))
# msno.bar(data)
# plt.show()
# 定义函数去掉当NAN大于50% 的列
file_name_new = '/dataset/{}.csv'.format('realtor_data_first1')

new_data = pd.DataFrame()

for column in data.columns:
    if len(data[column])!=0:
        column_is_null = pd.isnull(data[column])
        # print(column_is_null)
        column_is_null_true = column_is_null[column_is_null]
        # print(column_is_null_true)
        column_is_null_len = len(column_is_null_true)
        print(column, column_is_null_len)
        if column_is_null_len/len(data[column])<0.4:
            # print(data[column].dtype)
            new_data[column] = data[column]
            # print(new_data[column].dtype)

print('data_column_len:',len(data.columns))
print('new_data_column_len:',len(new_data.columns))
new_data.to_csv(current_path+ file_name_new,index=False)







## 单列的处理方式
# buildingTypeName_is_null = pd.isnull(data['buildingTypeName'])
# print(buildingTypeName_is_null)
# age_is_null_true = buildingTypeName_is_null[buildingTypeName_is_null]
# print(age_is_null_true)
# age_is_null_len = len(age_is_null_true)
# print(age_is_null_len)