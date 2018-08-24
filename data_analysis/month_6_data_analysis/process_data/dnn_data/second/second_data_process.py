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
data = data.drop(columns=['province','city','address','postalCode','kitchens'])
print('drop:',data.shape)










# 标签编码pandas 有自带的标签编码方法：
# def label_encode(data):
#     data_encode = pd.DataFrame()
#     encode_test = LabelEncoder()
#     for column in data.columns:
#         if data[column].dtype=='object':
#             data_encode_column = encode_test.fit_transform(np.array(data[column]).reshape(-1,1).flatten())
#         # print(data_class_encode)
#         else:
#             data_encode_column =list(data[column])
#         dataencode_column_dataframe = pd.DataFrame(data_encode_column,columns=[column])
#         data_encode = pd.concat((data_encode,dataencode_column_dataframe),axis=1)
#         print(column,data[column].dtype,data_encode.shape)
#
#     return data_encode
#
# data = label_encode(data)
# print(data.head())

def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            data[column] = data[column].astype('str')
    return data

data = label_encode(data)
print(data.head())
data['buildingTypeId'] = data['buildingTypeId'].astype('str')
print(data.dtypes)
print(data.shape)
# print(data.dtypes)
print(data.head())

# 看一下利用盒图去除离群点方法是否适用；
def remove_filers_with_boxplot(data):
    p = data.boxplot(return_type='dict')
    for index,value in enumerate(data[[x for x in data.columns if data[x].dtype !='object']].columns):
        if data[value].dtype!='object':
            # 获取异常值
            print(index,value)
            fliers_value_list = p['fliers'][index].get_ydata()
            print(fliers_value_list)
            # 删除异常值
            for flier in fliers_value_list:
                data = data[data.loc[:,value] != flier]
    return data

print(data.shape)
data = remove_filers_with_boxplot(data)
print(data.shape)



# 图形分析结果：
# 根据画图得到：parkingSpaces,不符合正太分布,分隔开的，可以分为三类；（0，1，2），（3，4），>=5
# bedrooms 也是不符合正太分布,分隔开的：可以分为三类：123，45，>=6
# longitude不太符合正太分布考虑删除小于-80的数据，根据图形所得小于-80的和大于-80的数据群有点分析；但是柱状图是连续的没有把柱形隔开；
# kitchen 的值全部为1，多以考虑将kitch剔除，但是这是在这个数据处理下的结果；其他的情况下不一定；
# washrooms也明显不符合正太分布，有三个部分的条状；，1，2），3，（4，5），>6=删除
# totalParkingSpaces:柱形图是连续的但是呈现起伏的形状，可能受极端值影响考虑剔除一些极端值，考虑剔除大于7的那一批数据；
# garageSpaces 看做成分类型数据：0，1，2，3
# data = data[data.longitude>-80]
data = data.drop(columns=['parkingSpaces','bedrooms','longitude','washrooms','totalParkingSpaces','garageSpaces'])
print(data.shape)
data.to_csv('./feature_select.csv',index=False)
# sns.pairplot(data[['latitude','longitude']])
# plt.show()


# sns.pairplot(data[[x for x in data.columns if data[x].dtype !='object']])
# plt.show()
# # 盒图测试bug,outofrange：主要是由于盒图获取离群点是只画连续型的数据
# p = data.boxplot(return_type='dict')
# print(len(data[[x for x in data.columns if data[x].dtype !='object']].columns))
# print(len(p['fliers']))
# # print(p['fliers'][27].get_ydata())
















'''
# 包裹型特征选择RFE;
from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
X = data.drop(columns='daysOnMarket')
Y = data['daysOnMarket']
names = list(data.columns)
names.remove('daysOnMarket')
lr = XGBRegressor()
rfe = RFE(lr, n_features_to_select = 20)
rfe.fit(X, Y)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 20), rfe.ranking_),names)))
print(rfe.n_features_to_select)

'''



'''
person相关性，corr()
# correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(100, 100))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

k = 40 #number of variables for heatmap
cols = corrmat.nlargest(k, 'daysOnMarket')['daysOnMarket'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''


'''
# 过滤型获取特征，评价分发：person相关系数，互信息，工业界使用少；
# 常用的方法：SelectKBest指定过滤个数，SelectPercentile指定过滤百分比；
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
'''

# 包裹型的方法：









# sns.pairplot(new_data)
# plt.show()

# 生成数据
# data.to_csv('./feature_select.csv',index=False)


