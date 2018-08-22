#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data_16000.py
@time: 2018/8/21
"""
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

data = pd.read_csv('./data_16000.csv')
data = data.dropna()
# data = data[[column for column in data.columns if data[column].dtype !='object']]
print(data.shape)


# msno.bar(data)


# print(data['bedrooms'].value_counts())
# print(data['city'].value_counts())

# city_sets = set(data['city'])
# counts = 0
# # 100:3600;10:2624:
# for city_set in city_sets:
#     if len(data[data.city==city_set]) >=300:
#         counts +=1
#
# print(counts)

# sns.pairplot(data)
# plt.show()


# 特征一个个的往上加，现在只考虑，缺失值少的那几个：
# 城市，地址，邮编，时间不考虑；
# 最终的特征数为： [['province','longitude',latitude','price','buildingTypeId','tradeTypeId','bedrooms','daysOnMarket']



data = data[['longitude','latitude','price','buildingTypeId','tradeTypeId','bedrooms','daysOnMarket']]
print(data.shape)
# sns.pairplot(data)
# plt.show()

# print(data['longitude'].value_counts())


# 处理province
# print(data['province'].value_counts()) # 发现只有一个省份Ontario ，模型应该分成若干省份；



# sortd_longitude = sorted(data['longitude'])
# print(sortd_longitude)
# data = data[data.longitude<-79]
# data = data[data.longitude>-80]
# print(data.shape)
#
# # 处理latitude
# sortd_latitude = sorted(data['latitude'])
# print(sortd_latitude)
# data = data[data.latitude>43]
#
#
# # 处理 price 这里的这里把交易类型分为两类；
# data = data[data.tradeTypeId==1]
#
#
# sortd_price= sorted(data['price'])
# print(sortd_price)
# data = data[data.price>90000]
# data = data[data.price<1000000]
# print(data.shape)
#
# # 处理buildingTypeId
# sortd_buildingTypeId= sorted(data['buildingTypeId'])
# print(sortd_buildingTypeId)
# print(data['buildingTypeId'].value_counts())
# data = data[data.buildingTypeId.isin([1,3,6])]
# print(data.shape)
#
#
# # 处理bedrooms
# print(data['bedrooms'].value_counts()) # 将大于6 的看成一类数据
# list_bedrooms_new = []
# for i in data['bedrooms']:
#     if i >6:
#         list_bedrooms_new.append(6)
#     else:
#         list_bedrooms_new.append(i)
# data['bedrooms'] = list_bedrooms_new
#
# print(data['bedrooms'].value_counts())
# print(data.shape)
#
# # 处理daysOnMarket
# print(data['daysOnMarket'].value_counts())
# sortd_daysOnMarket= sorted(data['daysOnMarket'])
# print(sortd_daysOnMarket)
# data = data[data.daysOnMarket<60]

# print(data.shape)
# data.to_csv('./little_columns1.csv',index=False)


# sns.pairplot(data)
# plt.show()


def same_processing_way(data):
    data = data[data.longitude < -79]
    data = data[data.longitude > -80]

    data = data[data.latitude > 43]

    data = data[data.tradeTypeId == 1]

    data = data[data.price > 90000]
    data = data[data.price < 1000000]

    data = data[data.buildingTypeId.isin([1, 3, 6])]

    list_bedrooms_new = []
    for i in data['bedrooms']:
        if i > 6:
            list_bedrooms_new.append(6)
        else:
            list_bedrooms_new.append(i)
    data['bedrooms'] = list_bedrooms_new

    data = data[data.daysOnMarket < 60]
    return data


