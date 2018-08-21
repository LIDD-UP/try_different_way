#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data_16000_add_column.py
@time: 2018/8/21
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

data = pd.read_csv('./data_16000.csv')
print(data.shape)

# data = data.dropna()

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

# data = same_processing_way(data)
data = data.dropna()
print(data.shape)

data = data.drop(columns=['province','city','address','postalCode','listingDate'])
# data_class = data[[column for column in data.columns if data[column].dtype =='object']]
# data_numeric = data[[column for column in data.columns if data[column].dtype !='object']]
# data_class = data_class.drop(columns=['province','city','address','postalCode','listingDate'])
#
# print(data_class.head())
# print(data_class.shape)
# for column in data_class.columns:
#     print(data_class[column].value_counts())


# 对所有的类别进行labelencoding；








from sklearn.preprocessing import LabelEncoder
import numpy as np



'''
调试concat的一个bug
encode_style = LabelEncoder()
style_encode = encode_style.fit_transform(data['style'])
style_ecode_datafram = pd.DataFrame(style_encode,columns=['style'])
print(style_ecode_datafram.shape)

# print(data['price'])
price = pd.DataFrame(list(data['price']),columns=['price'])
print(price.shape)
print(style_ecode_datafram.loc[0:10,:])
print(price.loc[0:10,:])

new_data1 = style_ecode_datafram.loc[0:10,:]
new_data2 = price.loc[0:10,:]

new_data3 = pd.concat((price,style_ecode_datafram),axis=1)

print(new_data3.shape)
'''





data_encode = pd.DataFrame()
encode_test = LabelEncoder()
for column in data.columns:
    if data[column].dtype=='object':
        data_encode_column = encode_test.fit_transform(np.array(data[column]).reshape(-1,1).flatten())
    # print(data_class_encode)
    else:
        data_encode_column =list(data[column])
    dataencode_column_dataframe = pd.DataFrame(data_encode_column,columns=[column])
    data_encode = pd.concat((data_encode,dataencode_column_dataframe),axis=1)
    print(column,data[column].dtype,data_encode.shape)

data_encode = data_encode.drop(columns='community')

print(data_encode.shape)

data_encode.to_csv('./process_data_16000_add_column.csv',index=False)


