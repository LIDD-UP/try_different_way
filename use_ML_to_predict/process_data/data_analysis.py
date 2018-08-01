#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/8/1
"""
import numpy as np
np.set_printoptions(suppress=False)
import pandas as pd
# pd.option_context('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from  sklearn.metrics import mean_absolute_error



# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

pd.set_option('max_columns',200)
pd.set_option('display.width',1000)






data_train_456 = pd.read_csv('../company_house_data/month_456_1.csv')
data_train_6 = pd.read_csv('../company_house_data/month_6_1.csv')
data_test_6 = pd.read_csv('../company_house_data/test_data_6_1.csv')
data_train_456 = data_train_456[pd.isna(data_train_456.buildingTypeId) !=True  ]
# print(data_train_456['buildingTypeId'].describe())
data_train_6 = data_train_6[pd.isna(data_train_6.buildingTypeId) !=True  ]
data_test_6 = data_test_6[pd.isna(data_test_6.buildingTypeId) !=True  ]
# 统计一下三种数据的省份城市和地址信息
def get_column_value_count(_data,_columns_list):
    for column in _columns_list:
        column_value_count = _data[column].value_counts()
        print('{} class counts is {}'.format(column,len(set(_data[column]))))
        print('{} value count-------------------------->>'.format(column))
        print(column_value_count)

# get_column_value_count(data_train_456,['province','city','address']) # 12,3470,116930
# get_column_value_count(data_train_6,['province','city','address']) # 11,1930,25290
# get_column_value_count(data_test_6,['province','city','address']) # 10, 277,843
# 通过以上统计可以看出除了province以外，其余的类别实在过多，基本上和数据量相同了，所以
# 先不考虑，等一下再尝试能不能用其他降维的方式再考虑对地址数据进行one_hot 编码；


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
data_train_456 = data_train_456[['longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
print('data_train_456 shape:',data_train_456.shape)
data_train_6 = data_train_6[['longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
print('data_train_6 shape:',data_train_6.shape)
data_test_6 = data_test_6[['longitude','latitude','price','buildingTypeId','bedrooms','daysOnMarket']]
print('data_test_6 shape:',data_test_6.shape)

# 取出label：
data_train_456_label = data_train_456['daysOnMarket']
data_train_6_label = data_train_6['daysOnMarket']
data_test_6_label = data_test_6['daysOnMarket']


'''
# # 把数据分割开后合并
# all_data_456 = pd.concat((data_train_456.loc[:,'longitude':'bedrooms'],data_test_6.loc[:,'longitude':'bedrooms']))
# all_data_6 = pd.concat((data_train_6.loc[:,'longitude':'bedrooms'],data_test_6.loc[:,'longitude':'bedrooms']))

# 画图查看log1p变换之前后之后的分布状态（最主要这种方法能将他转化成正太分布的形式）
# mpl.rcParams['figure.figsize'] = (12.0,6.0)
# # data_train_456_label_daysOnmarkets
# data_train_456_label_daysOnmarkets = pd.DataFrame({'daysOnmarket':data_train_456_label,'log(daysOnmarket+1）':np.log1p(data_train_456_label)})
# data_train_456_label_daysOnmarkets.hist()
#
# data_train_6_label_daysOnmarkets = pd.DataFrame({'daysOnmarket':data_train_6_label,'log(daysOnmarket+1）':np.log1p(data_train_6_label)})
# data_train_6_label_daysOnmarkets.hist()
#
# data_test_6_label_daysOnmarkets = pd.DataFrame({'daysOnmarket':data_test_6_label,'log(daysOnmarket+1）':np.log1p(data_test_6_label)})
# data_test_6_label_daysOnmarkets.hist()
#
# plt.show()


# 将label 进行 log1p变换
# data_train_456_label_log = np.log1p(data_train_456_label)
# data_train_6_label_log = np.log1p(data_train_6_label)
# data_test_6_label_log = np.log1p(data_test_6_label)


# # 将buildtypeid 转化成str类型的数；
# all_data['buildingTypeId'] = all_data['buildingTypeId'].astype(str)
# print('all_data shape:',all_data.shape)
# 
# # 通过log 变换skewed numeric features；
# # 获取数字特征的列
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# 
# # 从数字特征列中找到倾斜的特征列
# # # 找到data_train_456的
# # skewed_feats_456 =data_train_456[numeric_feats].apply(lambda x:skew(x.dropna()))
# # 
# # # 找到data_train_6中的
# # skewed_feats_6 =data_train_6[numeric_feats].apply(lambda x:skew(x.dropna()))
# def get_skewd_feature_to_log_transform(_data,numeric_feats):
#     # 计算倾斜特征的倾斜率
#     skew_feats = _data[numeric_feats].apply(lambda x:skew(x.dropna()))
#     # 获取倾斜率大于0.75的特征列
#     skewed_feats = skew_feats[skew_feats>0.75]
#     skew_feats = skewed_feats.index # 这是由于他是一个Series的结构，只有通过下标才能取得列名
#     return skew_feats
    
'''

# 把上面的处理过程全部分装成一个函数
def data_process(train,test,train_label,start_column,stop_column):
    all_data = pd.concat((train.loc[:, start_column:stop_column],
                          test.loc[:,start_column:stop_column]))

    all_data['buildingTypeId'] = all_data['buildingTypeId'].astype(str)
    print('all_data shape:',all_data.shape)

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
    return X_train,y,X_test


train,train_label,test = data_process(data_train_456, data_test_6, data_train_456_label, 'longitude','bedrooms')
print(train.head())
print(train.shape)
print(test.head())
print(test.shape)



'''
# model
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

# 交叉验证函数
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train_456, X_train_456_label, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
# 使用ridge 来进行预测
# model_ridge = Ridge(alpha=100)
# alphas = [x for x in np.arange(0,100,2)]
# alphas = np.logspace(4,100,50)
# print(alphas)
# cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
#             for alpha in alphas]
# cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation - Just Do It")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
#
# min_cv_ridge = cv_ridge.min() # 0.7768042324509667
# print(min_cv_ridge)
# plt.show() # 0.5
# model_ridge.fit(X_train_456,X_train_456_label)
# y_redge = np.expm1(model_ridge.predict(X_test_6))
# print(mean_absolute_error(data_test_6_label,y_redge)) # 9.6

# # 使用lasso来做
# from sklearn.linear_model import Lasso
# model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train_456, X_train_456_label)
# model_lasso = Lasso(alpha=0.01)
# cv_lasso = rmse_cv(model_lasso)
# cv_lasso = pd.Series(cv_lasso,index=[1, 0.1, 0.001, 0.0005,0.00001])
# cv_lasso.plot(title = "validation -just do it")
# plt.xlabel('alpha')
# plt.ylabel('rmse')
# plt.show()
# # mean:0.7768019817160832
# print(cv_lasso.min()) # 0.7281229082485158
# model_lasso.fit(X_train_456,X_train_456_label)
# y_redge = np.expm1(model_lasso.predict(X_test_6))
# print(mean_absolute_error(data_test_6_label,y_redge)) # 9.5744907479416

# 用 随机森林：
from sklearn.ensemble import RandomForestRegressor

# max_features = [.1,.3,.5,.7,.9,.99]
# test_scores = []
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=200,max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf,X_train_456,X_train_456_label,cv=5,scoring='neg_mean_squared_error'))
#     test_scores.append(test_score)
#
# plt.plot(max_features,test_scores)
# plt.title('alpha vs Error')
# plt.show()

# # 进行训练
# model_random_forest = RandomForestRegressor(n_estimators=500,max_features=.3)
# model_random_forest.fit(data_train_456,data_train_456_label)
# y_pr_forest = np.expm1(model_random_forest.predict(data_test_6))
# print(data_test_6.shape)
# print(mean_absolute_error(data_test_6_label,y_pr_forest))
'''

# 封装上面的函数：主要包括：交叉验证得分获取最佳参数（也就是调参的过程）
# 最后就是训练和预测的过程














