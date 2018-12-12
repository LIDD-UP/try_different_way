# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_quantile_based_buckets.py
@time: 2018/9/18
"""

import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


train_data = pd.read_csv('./input/month_67_trian_after_process_1.csv')
def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        "city",
        "province",
        "price",
        "tradeTypeId",
        # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
    ]]

    data = data.dropna(axis=0)
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list
    data = data.dropna(axis=0)
    return data


train_data = preprocess_data(train_data)



# msno.bar(train_data)
# plt.tight_layout()
# plt.show()

sns.pairplot(train_data)




plt.tight_layout()
plt.show()