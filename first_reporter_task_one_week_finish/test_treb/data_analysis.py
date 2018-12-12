# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_quantile_based_buckets.py
@time: 2018/10/23
"""
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

data_train = pd.read_csv('./input/treb_toronto_3to8.csv')
# data_train_all = pd.read_csv('./input/treb_all_column_month_3to8.csv')
#
# data_test = pd.read_csv('./input/treb_test_month_9.csv')
# data_test_all = pd.read_csv('./input/treb_test_all_column_month_9.csv')

# data = data_train.iloc[:,150:200]
msno.bar(data_train)
plt.tight_layout()
# plt.tick_params(labelsize=11)
plt.show()


'''
'province',
'city',
'address',
'postcode',
'longitude',
'latitude',
'price',
'buildingTypeId',
'tradeTypeId',
'listingDate',
'style',
'community',
'airConditioning',
'washrooms',
'basement1',
'familyRoom',
'fireplaceStove',
'heatSource',
'garageType',
'kitchens',
'parkingSpaces',
'parkingIncluded',
'rooms',
'waterIncluded',
'totalParkingSpaces',
'district',
'projectDaysOnMarket',
'daysOnMarket',
'bedrooms',
'ownerShipType',
'propertyTypeId',
'propertyType',
'bathroomTotal',
'treb',
'municpCode',
'communityCode',
'municipalityDistrict',
'municipality',
'pixUpdatedDate',
'remarksForClients',
'cableTVIncluded',
'cacIncluded',
'commonElementsIncluded',
'directions',
'extras',
'fireplaceStove',
'heatType',
'kitchens',
'parkingSpaces',
'parkingIncluded',
'listBrokerage',
'streetName',
'streetDirection',
'streetNo',
'streetAbbreviation',
'washroomsType1Pcs',
'washroomsType1',
'exterior1',
'privateEntrance',






'''

