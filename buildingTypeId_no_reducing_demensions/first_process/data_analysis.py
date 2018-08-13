#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/8/13
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('./input/month_6_1.csv')
test_data = pd.read_csv('./input/test_6_1.csv')

# dropna
train_data = train_data.dropna()
test_data = test_data.dropna()

train_data = train_data[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms','daysOnMarket']]
test_data = test_data[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms','daysOnMarket']]

# 打印column类型
print(train_data.dtypes)
print(test_data.dtypes)

# 打印形状
print(train_data.shape)
print(test_data.shape)

# 根据图形看出longitude存在严重的异常值，根据大于-3 去除
train_data = train_data[train_data.longitude < -3]
test_data = test_data[test_data.longitude <-3]

print(train_data.shape)
print(test_data.shape)

# 更具图形：latitude >40 ,latitude <58,price <90w,bedrooms <=8;
train_data = train_data[train_data.latitude>42]
train_data = train_data[train_data.latitude<58]
train_data = train_data[train_data.price<900000]
train_data = train_data[train_data.bedrooms<=8]
train_data = train_data[train_data.daysOnMarket<40]
train_data = train_data[train_data.longitude != train_data['longitude'].min()]

test_data = test_data[test_data.latitude>40]
test_data = test_data[test_data.latitude<58]
test_data = test_data[test_data.price<900000]
test_data = test_data[test_data.bedrooms<=8]
test_data = test_data[test_data.daysOnMarket<40]

print(train_data.shape)
print(test_data.shape)

train_data.to_csv('./month_6_train_1.csv',index=False)
test_data.to_csv('./test_data_1.csv',index=False)



# 现在利用盒图进行清理离群点；
















# sns.pairplot(train_data)
# # sns.pairplot(test_data)
# plt.show()

#




