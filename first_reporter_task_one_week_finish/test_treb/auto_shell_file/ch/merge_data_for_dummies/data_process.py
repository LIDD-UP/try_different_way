# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_process.py
@time: 2018/11/28
"""
import pandas as pd


data = pd.read_csv('./solve_dummies_problem.csv')

data = data[[
            "longitude",
            "latitude",
            # "city",
            # "province",
            "price",
            "tradeTypeId",
            # # "listingDate",
            "buildingTypeId",
            "bedrooms",
            "bathroomTotal",
            # 'postalCode',
            'daysOnMarket',
            'ownerShipType',
            # 'projectDaysOnMarket',
            'district',
]]
data = data.dropna()
new_data = pd.DataFrame()
print(data.shape)
for column in data.columns:
    if column in [
            "tradeTypeId",
            "buildingTypeId",
            'ownerShipType',
            'district']:
        new_data = pd.concat((pd.DataFrame(list(set(list(data[column])))),new_data),axis=1)
#

print(new_data.shape)
print(new_data)

# buildingTypeId = pd.DataFrame(list(set(list(data['buildingTypeId']))))
# district = pd.DataFrame(list(set(list(data['district']))))
# print(buildingTypeId)
# print(district.shape)
#
# merge_data = pd.concat((district,buildingTypeId),axis=1)
# print(merge_data.shape)
# print(merge_data)