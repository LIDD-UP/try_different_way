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
from scipy.stats import boxcox

train_data = pd.read_csv('../input/month_6_1.csv')
test_data = pd.read_csv('../input/test_6_1.csv')


# dropna
train_data = train_data.dropna()
test_data = test_data.dropna()

train_data['bedrooms'] = train_data['bedrooms'].astype(int)
test_data['bedrooms'] = test_data['bedrooms'].astype(int)

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
train_data = train_data[train_data.buildingTypeId.isin([3,1,6,19,12,17,13,7,16,14])]
train_data = train_data[train_data.bedrooms.isin([0,1,2,3,4,5,6,7])]

test_data = test_data[test_data.latitude>40]
test_data = test_data[test_data.latitude<58]
test_data = test_data[test_data.price<900000]
test_data = test_data[test_data.bedrooms<=8]
test_data = test_data[test_data.daysOnMarket<40]
test_data = test_data[test_data.buildingTypeId.isin([3,1,6,19,12,17,13,7,16,14])]
test_data = test_data[test_data.bedrooms.isin([0,1,2,3,4,5,6,7])]

print(train_data.shape)
print(test_data.shape)


# 利用盒图去除离群点，只在price，longitude，latitude，daysonmarket中考虑；
# 观测效果并不是特别理想；去掉daysonmarkt之后要好一点；
def remove_filers_with_boxplot(data):
    p = data.boxplot(return_type='dict')
    for index,value in enumerate(['longitude','latitude','price']):
        # 获取异常值
        fliers_value_list = p['fliers'][index].get_ydata()
        # 删除异常值
        for flier in fliers_value_list:
            data = data[data.loc[:,value] != flier]
    return data


# train_data = remove_filers_with_boxplot(train_data)
# print(train_data.shape)

# 根据分类来去除离群点；
def use_pivot_box_to_remove_fliers(data,pivot_columns_list,pivot_value_list):
    for column in pivot_columns_list:
        for value in pivot_value_list:
            # 获取分组的dataframe
            new_data = data.pivot(columns=column,values=value)
            p = new_data.boxplot(return_type='dict')
            for index,value_new in enumerate(new_data.columns):
                # 获取异常值
                fliers_value_list = p['fliers'][index].get_ydata()
                # 删除异常值
                for flier in fliers_value_list:
                    data = data[data.loc[:, value] != flier]
    return data

train_data['buildingTypeId'] = train_data['buildingTypeId'].astype(str)

print(train_data.dtypes)
print(train_data['buildingTypeId'].value_counts())
print(test_data['buildingTypeId'].value_counts())
train_data['bedrooms'] = train_data['bedrooms'].astype(str)
print(train_data['bedrooms'].value_counts())
print(test_data['bedrooms'].value_counts())

trian_data = use_pivot_box_to_remove_fliers(train_data,['buildingTypeId','bedrooms'],['longitude','latitude','price','daysOnMarket'])
print(train_data.shape)

# train_data.to_csv('./month_6_train_1.csv',index=False)
# test_data.to_csv('./test_data_1.csv',index=False)
train_data = train_data.dropna()

# train_data['longitude'] = abs(train_data['longitude'])
# train_data['longitude'] = np.log1p(np.log1p(train_data['longitude']))

print(train_data.shape)




# 现在利用盒图进行清理离群点；
















# sns.pairplot(train_data)
# # sns.pairplot(test_data)
# plt.show()

#




