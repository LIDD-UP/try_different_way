# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: second_data_process.py
@time: 2018/8/23
"""
import numpy as np
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./second.csv')
print(data.shape)
print(data.head())

print(len(set(data['city'])))
print(len(set(data['postalCode'])))


# 去掉类别过多的列：
data = data.drop(columns=['province','city','address','postalCode'])
print('drop:',data.shape)


# 标签编码
def label_encode(data):
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

    return data_encode

data = label_encode(data)
print(data.shape)
print(data.dtypes)
print(data.head())

#correlation matrix
# corrmat = data.corr()
# f, ax = plt.subplots(figsize=(100, 100))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

# k = 40 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'daysOnMarket')['daysOnMarket'].index
# cm = np.corrcoef(data[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data['longitude'] = abs(data['longitude'])
X = data.drop(columns='daysOnMarket')
y= data['daysOnMarket']
model = SelectKBest(chi2, k=20)
X_new = model.fit_transform(X, y)




print(X_new.shape)
X_new = pd.DataFrame(X_new,columns=[ 'y'+str(x) for x in range(20)])

print(X_new.head())
new_data = pd.concat((X_new,data['daysOnMarket']),axis=1)
print(new_data.head())




new_data['price'] = new_data['y0']
new_data = new_data.drop(columns='y0')
print(new_data.head())

print(data[data.daysOnMarket>80].head())

# sns.pairplot(new_data)
# plt.show()


# 生成数据
# new_data.to_csv('./feature_select.csv',index=False)


