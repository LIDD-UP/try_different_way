# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: datanaalysis.py
@time: 2018/9/13
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice
import  missingno as msno
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./input/train_month_7_d.csv')
print(data.head())
print(data.shape)
data_test = pd.read_csv('./input/test.csv')
print(data_test.shape)
data_month_8_delisting = pd.read_csv('./input/month_8_data_delisting.csv')
print(data_month_8_delisting.shape)


# def data_process(data):
#     data = data.drop(columns=['id','address','airConditioning','sizeExterior','sizeInterior','storiesTotal'])
#     data = data.dropna()
#
#
#     # data = data[[column for column in data.columns if data[column].dtype!='object']]
#     bedrooms_list = []
#     for bedroom in data['bedrooms']:
#         bedrooms_list.append(int(eval(bedroom)))
#     data['bedrooms'] = bedrooms_list
#
#     listingDateMonth = []
#     for item in data["listingDate"]:
#         # print(item)
#         if '-' in item:
#             listingDateMonth.append(int(item.split('-')[1]))
#         else:
#             listingDateMonth.append(int(item.split('/')[1]))
#     data["listingDataMonth"] = listingDateMonth
#     return data
#
#
# data = data_process(data)
#
# print(data.head())
#
# sns.pairplot(data)
# plt.tight_layout()
# plt.show()


# msno.bar(data)
# plt.tight_layout()
# plt.show()
# data = data[[
#     "longitude",
#         "latitude",
#         "city",
#         "province",
#         "price",
#         # "propertyType",
#         "tradeTypeId",
#         "listingDate",
#         "buildingTypeId",
#         "bedrooms",
#         "bathroomTotal",
#         'daysOnMarket'
#          ]]

# data_corr = data.corr()
#
# sns.heatmap(data_corr,annot=True)
# plt.tight_layout()
# plt.show()