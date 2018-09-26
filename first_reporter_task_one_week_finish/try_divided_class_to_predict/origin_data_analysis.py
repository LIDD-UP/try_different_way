# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: origin_data_analysis.py
@time: 2018/9/26
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



train_data = pd.read_csv('./input/month_567_data.csv')
test_data = pd.read_csv('./input/hose_info_201808_predict_2.csv')

print('train:',train_data.shape)
print('test shape',test_data.shape)

def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        "city",
        "province",
        "price",
        "tradeTypeId",
        "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        'postalCode',
        'daysOnMarket',
        'ownerShipType'
    ]]
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    print('data shape=%s before dropna' % (str(data.shape)))
    data = data.dropna(axis=0)
    print('data shape=%s after dropna' % (str(data.shape)))
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if isinstance(bedrooms, float):
            bedrooms_list.append(int(bedrooms))
        else:
            bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list
    return data


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# corr = train_data.corr('spearman')
# sns.heatmap(corr,annot=True)
# plt.tight_layout()
# plt.show()


print(train_data.head())
print(test_data.head())

category_variable = [
                        # 'province',
#                      'city',
#                      'tradeTypeId',
#                      'buildingTypeId',
                    'price',
                     'bedrooms',
                     # 'bathroomTotal',
                     # 'postalCode',
                     # 'ownerShipType',
                     # 'year', 'month',
                     # 'daysOnMarket'
                     ]

# print(train_data['province'].value_counts())


# sns.pairplot(train_data)
# plt.tight_layout()
# plt.show()


def use_std_to_remove_fliers(data,column):
    outliers_collections = []
    price_mean = data[column].mean()
    price_std = data[column].std()
    for index,value in enumerate(data[column]):
        if abs(value-price_mean)>3*price_std:
            outliers_collections.append(value)
    return outliers_collections

def data_process(data):
    # data= data[data.daysOnMarket]
    data = data[data.bedrooms<1000]

    return data

# price 去除离散值
outliers_collections = use_std_to_remove_fliers(train_data)
print(outliers_collections)
print(len(outliers_collections))
train_data = train_data[~train_data.price.isin(outliers_collections)]

# outliers_collections = use_std_to_remove_fliers(train_data)
# print(outliers_collections)
# print(len(outliers_collections))
# train_data = train_data[~train_data.price.isin(outliers_collections)]

# daysOnMarket 去除离散值
outliers_collections = use_std_to_remove_fliers(train_data)
print(outliers_collections)
print(len(outliers_collections))
train_data = train_data[~train_data.price.isin(outliers_collections)]




sns.pairplot(train_data)
plt.tight_layout()
plt.show()