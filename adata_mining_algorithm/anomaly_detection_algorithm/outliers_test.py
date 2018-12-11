#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: standard_process.py
@time: 2018/8/9
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
'''
通过box_plot(盒图来确认）异常值
'''

# 获取项目根目录
input_data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/input/'
print(input_data_path)

# 获取数据得位置
month_6_train_path = input_data_path +'month_6_1.csv'
month_6_test_path = input_data_path + 'test_data_6_1.csv'

# 读取数据
data_train = pd.read_csv(month_6_train_path)
data_test = pd.read_csv(month_6_test_path)

# print(data_train.head())
# print(data_test.head())

# 暂时不考虑省份城市地址
# 月份只有一个月，暂时不考虑
# bedrooms 需要看成分类型得数据
# 只取出longitude，latitude，price，buildingTypeId,bedrooms,daysOnMarket


# 取出这些数据；
# train = data_train[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
# train= train.dropna()
train = data_test[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print(train.head())
# print(test.head())
# print(train.isna().sum())
# sns.pairplot(train)
# # sns.pairplot(test)
# plt.show()


# 特征清洗：异常值清理用用箱图；
# 分为两步走，一步是单列异常值处理，
# 第二步是多列分组异常值处理
def remove_filers_with_boxplot(data):
    p = data.boxplot(return_type='dict')
    for index,value in enumerate(data.columns):
        # 获取异常值
        fliers_value_list = p['fliers'][index].get_ydata()
        # 删除异常值
        for flier in fliers_value_list:
            data = data[data.loc[:,value] != flier]
    return data

print(train.shape)
train = remove_filers_with_boxplot(train)
print(train.shape)

'''
以上得异常值处理还不够完善，
完善的异常值处理是分组判断异常值，
也就是他在单独这一列种,还有一种情况是多余不同的分类，他是不是存在异常
所以就需要用到分组获取数据再箱图处理掉异常数据；
'''
train = train[pd.isna(train.buildingTypeId) != True]
print(train.shape)

print(train['bedrooms'].value_counts())
'''
3.0    8760
2.0    5791
4.0    5442
1.0    2056
5.0    1828
6.0     429
0.0     159
7.0      82
由于样本存在不均衡得问题：所以只采用12345数据：也就是说去掉0，7，6，到时候测试数据也要做相同得操作；
还有一种是通过下采样或者是上采样的方式进行，这里暂时不考虑；
'''
# 只取bedrooms 为1，2，3，4，5 得数据
train = train[train['bedrooms'].isin([1,2,3,4,5])]
print(train.shape)


# 利用pivot分组后去掉异常点
def use_pivot_box_to_remove_fliers(data,pivot_columns_list,pivot_value_list):
    for column in pivot_columns_list:
        for value in pivot_value_list:
            # 获取分组的dataframe
            new_data = data.pivot(columns=column,values=value)
            p = new_data.boxplot(return_type='dict')
            for index,value_new in enumerate(new_data.columns):
                # 获取异常值
                fliers_value_list = p['fliers'][index].get_ydata()
                # 删除异常值
                for flier in fliers_value_list:
                    data = data[data.loc[:, value] != flier]
    return data


# train = use_pivot_box_to_remove_fliers(train,['buildingTypeId','bedrooms'],['price','daysOnMarket','longitude','latitude'])
print(train.shape)
# print(train.isna().sum())

# 以上就不考虑longitude和latitude的问题了；应为房屋的类型以及房间个数和经纬度关系不大,但是也不一定，
# 实践了一下加上longitude和latitude之后样本数据并没有减少；

# sns.pairplot(train)
# plt.show()

# 先进一步做处理将纬度小于40的去掉
train = train[train.latitude>40]

# --------------------------------》》》
# 对于数值类型得用均值填充，但是在填充之前注意一些原本就是分类型数据得列
# def fill_na(data):
#     for column in data.columns:
#         if column.dtype != str:
#             data[column].fillna(data[column].mean())
#     return data

# 以上是异常值，或者是离群点的处理，以及均值填充数据
# 下面将根据catter图或者是hist图来处理数据


# # 标准化数据
# train = StandardScaler().fit_transform(train)
# # 标准化之后画图发现数据分布并没有变
#
# sns.pairplot(pd.DataFrame(train))
# plt.show()

'''
1:循环遍历整个散点图用刚才写好的算法去除点；
'''

# 获取
# def get_outlier(x,y,init_point_count ,distance,least_point_count):
#     x_outliers_list = []
#     y_outliers_list = []
#     for i in range(len(x)):
#         for j in range(len(x)):
#              d =np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
#              # print('距离',d)
#              if d <= distance:
#                 init_point_count +=1
#         if init_point_count <least_point_count+1:
#             x_outliers_list.append(x[i])
#             y_outliers_list.append(y[i])
#             print(x[i],y[i])
#         init_point_count =0
#     return x_outliers_list,y_outliers_list
#
# def circulation_to_remove_outliers(data,list_columns=['longitude','latitude','price','daysOnMarket',]):
#     for column_row in list_columns:
#         for column_col in list_columns:
#             if column_row != column_col:
#                 x = list(data[column_row])
#                 y = list(data[column_col])
#                 x_outliers_list ,y_outliers_list = get_outlier(x,y,0,0.01,2)
#                 for x_outlier in x_outliers_list:
#                     data = data[data.loc[:, column_row] != x_outlier]
#                 for y_outlier in y_outliers_list:
#                     data = data[data.loc[:, column_col] != y_outlier]
#     return data
#
# train = circulation_to_remove_outliers(train)
#
# print(train.shape)






# def get_outlier(x,y,init_point_count ,distance,least_point_count):
#     for i in range(len(x)):
#         for j in range(len(x)):
#              d =np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
#              # print('距离',d)
#              if d <= distance:
#                 init_point_count +=1
#         if init_point_count <least_point_count+1:
#             print(x[i],y[i])
#         init_point_count =0
#
# get_outlier(train['longitude'],train['latitude'],0,0.3,1)








# sns.pairplot(train)
# plt.show()
# train = train.dropna()
# print(train.tail())
# train.to_csv('./finnl_processing_train_data_6_no_remove_outliers_test.csv',index=False)











