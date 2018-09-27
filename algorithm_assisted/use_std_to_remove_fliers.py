# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: use_std_to_remove_fliers.py
@time: 2018/9/27
"""
def use_std_to_remove_fliers(data,column):
    # 必须重置下标之后才能用下标的方式获取fliers的信息：
    print(data.shape)
    data = data.reset_index(drop=True) # drop =True可以删除原来行的索引
    outliers_collections = []
    column_mean = data[column].mean()
    column_std = data[column].std()
    for index,value in enumerate(data[column]):
        if abs(value-column_mean)>3*column_std:
            outliers_collections.append(index)
    data = data[~data.index.isin(outliers_collections)]
    print(data.shape)
    return data