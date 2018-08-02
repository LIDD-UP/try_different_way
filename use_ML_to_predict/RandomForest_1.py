#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: RandomForest_1.py
@time: 2018/8/1
"""
import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

data_train_456 = pd.read_csv('./company_house_data/month_456_1.csv')
data_train_6 = pd.read_csv('./company_house_data/month_6_1.csv')
data_test_6 = pd.read_csv('./company_house_data/test_data_6_1.csv')

# 去掉buildingTypeId 为空的情况避免再编码的时候出现na这一类
data_train_456 = data_train_456[pd.isna(data_train_456.buildingTypeId) != True]
data_train_6 = data_train_6[pd.isna(data_train_6.buildingTypeId) != True]
data_test_6 = data_test_6[pd.isna(data_test_6.buildingTypeId) != True]


# 统计 count(不包括缺失值的情况）
# print(data_train_456.describe())
# print(data_train_6.describe())
# print(data_test_6.describe())
# 通过查看发现bedrooms 最大值和最小值差距有点大需要用value_counts查看一离群点；
# 接下来要对所有列进行离群点，缺失值，查看；

# data_train_456['price'].hist()
# np.log1p(data_train_456['daysOnMarket']).hist()
# plt.show()


# 接下来处理步骤：
# 1：去掉省份城市地址；
# 2：将原始数据的列调整顺序；
# 3：然后将训练数据和测试数据的标签数据和特征数据分开；
# 4：把训练数据和测试数据组合起来cancat，一起进行处理，最后再通过index取出来；
# 5: 将原本的数据中本身是类别的buildingtype转换成str，
# 6: 将原本的数值数据进行log1p处理
# 7：对buildingtypeid 进行one_hot 编码
# 8：填充缺失值
# 9：获取训练数据测试数据的最后数据；
# 10 建模处理


# 去掉省份城市地址，调整顺序
data_train_456 = data_train_456[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_train_456 shape:', data_train_456.shape)
data_train_6 = data_train_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_train_6 shape:', data_train_6.shape)
data_test_6 = data_test_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_test_6 shape:', data_test_6.shape)

# 取出label：
data_train_456_label = data_train_456['daysOnMarket']
data_train_6_label = data_train_6['daysOnMarket']
data_test_6_label = data_test_6['daysOnMarket']


# 数据处理过程
def data_process(train, test, train_label, start_column, stop_column):
    all_data = pd.concat((train.loc[:, start_column:stop_column],
                          test.loc[:, start_column:stop_column]))

    all_data['buildingTypeId'] = all_data['buildingTypeId'].astype(str)
    print('all_data shape:', all_data.shape)

    train_label = np.log1p(train_label)

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())
    all_data = pd.get_dummies(all_data)

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train_label
    return X_train, y, X_test

# 获取处理之后的数据

# 获取train_456 的数据
train, train_label, test = data_process(data_train_456, data_test_6, data_train_456_label, 'longitude', 'bedrooms')

# 获取trian_6的数据
# train, train_label, test = data_process(data_train_6, data_test_6, data_train_6_label, 'longitude', 'bedrooms')

# print(train.head())
# print(train.shape)
# print(test.head())
# print(test.shape)








# 用 随机森林：
from sklearn.ensemble import RandomForestRegressor

# 交叉验证
# max_features = [.1,.3,.5,.7,.9,.99]
# test_scores = []
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=200,max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf,train,train_label,cv=5,scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# plt.plot(max_features,test_scores)
# plt.title('RandomForest alpha vs Error')
# plt.show()

# 进行训练
model_random_forest = RandomForestRegressor(n_estimators=200,max_depth=8)
model_random_forest.fit(train,train_label)
y_pr_forest = np.expm1(model_random_forest.predict(test))
print(test.shape)
print(mean_absolute_error(data_test_6_label,y_pr_forest))

plt.plot(y_pr_forest[0:100],c='red',label="pre")
plt.plot(data_test_6_label[0:100],c='black',label='true')
plt.title("RandomForest_1 pre and label distribute circumstance")
plt.legend()
plt.show()
# 2.5784470987831304e+19
