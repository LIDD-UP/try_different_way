#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: dnn_data_process.py
@time: 2018/8/22
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./month6.csv')

# 先去除掉之前观测出来的无用特征：
data = data.drop(columns=['id',
                          'listingDate','basement1','cableTVIncluded',
                          'heatType', 'room5','streetDirection','room5Length','room5Width',
                          'cacIncluded'
                          ])
# 处理bedrooms
def process_bedrooms(data):
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        if i != 'nan':
            list_month_process.append(eval(i))
        else:
            list_month_process.append(i)
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('float')
    return data


data = process_bedrooms(data)

# 分开tradetype
data = data[data.tradeTypeId == 1]


# 观测缺失值
# msno.bar(data)
# plt.show()

# 检测不同数据量的保留，特征的保留情况
def test_columns_num_with_diff_data(data,data_num):
    columns_set = []
    for column in data.columns:
        if len(data[pd.notna(data[column]) ])>data_num:
            columns_set.append(column)
    data = data[columns_set]
    return data,len(columns_set)
# data,len_columns = test_columns_num_with_diff_data(data,10000)


def check_dropna_before_after(data):
    print(data.shape)
    print(data.head())
    data = data.dropna()
    print(data.shape)
    print(data.head())



# sns.pairplot(data)
# plt.show()

# 现在一个特征一个特征的网上添加：在此之前要先确定特征的处理方式
# 需不需要进行归一化或者是标准化操作；


# 第一步：拿以前的来测试，[['price','buildingTypeId']]
data = data[['province','city','address','postalCode','longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
data['buildingTypeId'] = data['buildingTypeId'].astype(str)
print(data.shape)
print(data.dtypes)
def same_processing_way(data):
    data = data[data.longitude < -10]
    data = data[data.longitude > -140]
    #
    data = data[data.latitude > 43]
    # data = data[data.tradeTypeId == 1]
    #
    data = data[data.price > 50000]
    data = data[data.price < 2000000]
    #
    # data = data[data.buildingTypeId.isin([1, 3, 6])]
    #
    # list_bedrooms_new = []
    # for i in data['bedrooms']:
    #     if i > 6:
    #         list_bedrooms_new.append(6)
    #     else:
    #         list_bedrooms_new.append(i)
    # data['bedrooms'] = list_bedrooms_new
    #
    # data = data[data.daysOnMarket < 60]
    return data
data = same_processing_way(data)

print(data.shape)


data.to_csv('./dnn_data/first.csv',index=False)



# sns.pairplot(data)
# plt.show()
