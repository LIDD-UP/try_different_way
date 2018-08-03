#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: Bagging.py
@time: 2018/8/3
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


# 获取train_456 的数据
train, train_label, test = data_process(data_train_456, data_test_6, data_train_456_label, 'longitude', 'bedrooms')

from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

# 寻找超参数
params_linear ={"alpha":np.logspace(-5,3,1000)}
model_linears = [Ridge(),Lasso()]
model_best = []
for i in model_linears:
    linear_grid = GridSearchCV(estimator=i,param_grid=params_linear,scoring='neg_mean_absolute_error')
    linear_grid.fit(train,train_label)
    model_best.append(linear_grid.best_estimator_)





'''
base_estimator=None,
n_estimators=10,
max_samples=1.0,
max_features=1.0,
'''
params = {'base_estimator':model_best,"max_samples":[x for x in range(2,3)],"n_estimators":[x for x in range(1,1000,1)],'max_features':[x for x in range(1,2,1)]}
grid = GridSearchCV(estimator=BaggingRegressor(),param_grid=params,scoring='neg_mean_absolute_error')

# 训练
grid.fit(train,train_label)
# print(len(grid.cv_results_.values()))
# print(help(grid.))
# 打印最好参数和最好的得分值
print('best_params',grid.best_params_)
print('best_scoring',grid.best_score_)
model = grid.best_estimator_
print(model)

# 预测
preds = np.expm1(model.predict(test))
print('error',mean_absolute_error(data_test_6_label,preds))
print('pred_mean',preds.mean())
print('true_mean',data_test_6_label.mean())

# 画图
plt.figure(figsize=(100,100))
plt.plot(preds[0:100],c='blue',label='pred')
plt.plot(data_test_6_label,c='red',label='true')
plt.title("RandomForest preds and true daysOnMarket distribution circumstance")
plt.legend()
plt.show()
'''
error 9.711804250253497
pred_mean 21.53896336091896
true_mean 18.813953488372093
'''




