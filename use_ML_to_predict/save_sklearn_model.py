#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: save_sklearn_model.py
@time: 2018/8/2
"""
'''
机器学习算法主要有两种方式：保存模型（sklearn下）
第一种是通过sklearn下得externals joblib得方式；




'''
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



from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

# 交叉验证

# 训练：及保存模型：joblib更适合大数据量得模型，且只能往硬盘存储，不能往字符串存储；
random_forest = RandomForestRegressor(n_estimators=400,max_depth=6)
random_forest.fit(train,train_label)
# joblib.dump(random_forest,"train_model.m")

# 导入模型预测：
# model = joblib.load('train_model.m')
# preds = model.predict(test)
# print(mean_absolute_error(data_test_6_label,preds))
# 15.8135039127917


# 方法二 ，通过pickle 模块来保存sklearn 内部得joblib；
# import pickle
# with open('save/clf.pickle','wb') as f:
#     pickle.dump(random_forest,f)


# with open('save/clf.pickle','rb') as f:
#     model = pickle.load(f)
#     preds = model.predict(test)
#     print(mean_absolute_error(data_test_6_label,preds))
