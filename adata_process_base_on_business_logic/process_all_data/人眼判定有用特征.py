#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: 人眼判定有用特征.py
@time: 2018/7/13
"""

import pandas as pd
import os
import re



def read_csv_data(filename):
    # 这种方式的文件读取方式仅限于py文件和dataset 文件属于同一级目录下
    current_path = os.getcwd() #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    data = pd.read_csv(file_path)
    return data


def to_csv_file(dataframe,filename):
    # 把dataframe转换成csv文件并存储到dataset下,仅限于py文件和dataset文件属于同一级目录下
    current_path = os.getcwd()  #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    dataframe.to_csv(file_path, index=False)

feature_import = '''
address,province, city, price, 
buildingTypeId,tradeTypeId ,listingDate ,furnished,
approxSquareFootage, trebStatus ,style ,municpCode,
communityCode, municipalityDistrict ,municipality ,
community,airConditioning ,acreage,washrooms ,bedrooms ,
bedroomsPlus,basement1,basement2 ,cableTVIncluded,cacIncluded ,
commonElementsIncluded  ,frontingOn  ,directions,familyRoom ,lotDepth,
drive ,utilitiesHydro,extras, fireplaceStove ,lotFront, 
heatSource,garageType ,utilitiesGas ,heatType, lotIrregularities ,
kitchens, parkingSpaces,pool,parkingIncluded ,listBrokerage,
room1Length,room1,room1Width,room2Length,room2,room2Width,
room3Length,room3,room3Width,room4Length,room4,room4Width,
room5Length,room5,room5Width,room6Length,room6,room6Width,
room7Length,room7,room7Width,room8Length,room8,room8Width,
room9Length,room9,room9Width,rooms,sewers,streetName,
streetDirection,streetNo,taxes,uffi,waterIncluded,
washroomsType1Pcs,washroomsType2Pcs,washroomsType3Pcs,
washroomsType4Pcs,washroomsType1,washroomsType2,washroomsType3,
washroomsType4,taxYear,approxAge,zoning,typeOwnSrch,typeOwn1Out,
exterior1,exterior2,otherStructures1,garageSpaces,laundryAccess,
privateEntrance,kitchensPlus,laundryLevel,propertyFeatures3,propertyFeatures4,
propertyFeatures5,propertyFeatures6,retirement,waterfront,specialDesignation1,
parcelOfTiedLand,totalParkingSpaces,district,latitude,longitude
'''
a_list_true =[]
a_list_1 = feature_import.split(',')
for i in a_list_1:
    i = i.strip()
    i = i.replace('\n','')
    # print(i)
    if i == 'price':
        print('find')
    a_list_true.append(i)
# print(a_list_true)
import_feature_list = a_list_true
# 经过大于10000有效数据到正则，最后人眼观测；的到的数据：
data = read_csv_data('re_delete')
new_data = pd.DataFrame()
for column in import_feature_list:
    new_data[column] = data[column]
print(new_data.shape)
to_csv_file(new_data,'human_judge')

