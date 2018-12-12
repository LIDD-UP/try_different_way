#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: delete_rows.py
@time: 2018/7/12
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import missingno as msno

current_path = os.getcwd()
fil_name = '/dataset/{}.csv'.format('test_house_info')
file_path = current_path + fil_name
data = pd.read_csv(file_path)
# 定义新文件存储位置
# file_name_new = '/dataset/{}.csv'.format('test_house_info')

# 尝试函数
# print(data.loc[])
# print(len(data))
new_data = pd.DataFrame()
count = 0 # 第几次删除行，第一次就要用原始数据进行赋值，后面就不用了，直接inplace=True，为了保留原始文件；
for i in range(len(data)):

    # print(data.loc[i])
    row_i_len = len(data.loc[i])
    print('row_i_len',row_i_len)
    if row_i_len !=0:
        row_i = data.loc[i]
        print('具体位置：',row_i['price'])
        if pd.isna(row_i['price']):
            print('是的')
        # print('row_i',row_i)
        # 获取每行缺失值个数
        row_i_is_null = pd.isnull(row_i)
        row_i_is_null_true = row_i_is_null[row_i_is_null]
        row_i_is_null_len = len(row_i_is_null_true)
        print('row_i_is_null_len',row_i_is_null_len)
        # 若行的缺失值大于等于50% 就删除掉
        if row_i_is_null_len /row_i_len >=0.5:
            count += 1
            if count==1:
                new_data = data.drop(index=i)
                print('new_data_len_i',len(new_data))
            if count>1:
                new_data.drop(index=i,inplace=True)
    # if i == 3:
    #     break

print('data_len',len(data))
print('new_data_len_finnal',len(new_data))

# new_data.to_csv(current_path+ file_name_new,index=False)

