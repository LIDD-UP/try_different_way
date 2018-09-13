# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/9/12
"""
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

data = pd.read_csv('../input/half_a_year.csv')

# msno.bar(data)
# plt.tight_layout()
# plt.show()

# postCodeList = []
# for item in data["postalCode"]:
#     print(item)
#     print(item.split(' ')[0])
#     postCodeList.append(item.split(' ')[0])
# data["postalCodeThreeStr"] = postCodeList


def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i + 1.) / (num_buckets + 1.) for i in range(num_buckets)])
    print(quantiles)
    return [quantiles[q] for q in quantiles.keys()]

x = get_quantile_based_buckets(data['price'],60)
print(x)





