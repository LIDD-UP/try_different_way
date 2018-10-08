# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: origin_data_analysis.py
@time: 2018/9/26
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek,SMOTEENN



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


# price 去除离散值
# train_data = use_std_to_remove_fliers(train_data,'price')
# train_data = use_std_to_remove_fliers(train_data,'daysOnMarket')


def data_process(data):
    # data= data[data.daysOnMarket]
    data = data[data.bathroomTotal<1000]
    # data = data[data.bedrooms<15]
    # print(data['buildingTypeId'].value_counts())
    # data = data[~data.buildingTypeId.isin([14,18,17,16,13,10,2,7,5])]
    # print(data['buildingTypeId'].value_counts())
    # print(data['province'].value_counts())
    # data = data[~data.province.isin(['Newfoundland & Labrador','Yukon','New Brunswick'])]
    # print(data['province'].value_counts())

    return data


train_data = data_process(train_data)

print(train_data.shape)
train_data.to_csv('./train_process_price.csv',index=False)





# sns.pairplot(train_data)
# plt.tight_layout()
# plt.show()