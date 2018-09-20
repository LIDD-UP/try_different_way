# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: xgboost_to_predict.py
@time: 2018/9/20
"""
# -*- coding:utf-8 _*-
""" 
@author:Administrator
@file: xgboost_template.py
@time: 2018/9/20
"""
import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV,KFold

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

# 1：数据读取
train_data = pd.read_csv('./input/month_7_delisting_after_process_2.csv')
train_data = train_data.drop(columns=['记录数'])

test_data = pd.read_csv('./input/hose_info_201808_predict.csv')

# 2：数据处理
def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        # "city",
        "province",
        "price",
        "tradeTypeId",
        # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType'
    ]]
    # 只取出买卖类型未出售的数据;这个字段待定针对这批预测数据;
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])

    data = data.dropna(axis=0)
    # 重置下标,删除index列
    data = data.reset_index()
    data = data.drop(columns=['index'])
    origin_data = data
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if pd.isna(bedrooms) == True:
            bedrooms_list.append(bedrooms)
        elif isinstance(bedrooms,float):
            bedrooms_list.append(int(bedrooms))
        else:
            bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        if pd.isna(bathroom_total) == True:
            bathroom_total_list.append(bathroom_total)
        else:
            bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list

    return data,origin_data


train_data,origin_train_data= preprocess_data(train_data)
test_data, origin_data = preprocess_data(test_data)


# 3：特征变换，log或者其他的方式
# log变换price
# train_data['price'] = np.log1p(train_data['price'])
# test_data['price'] = np.log1p(test_data['price'])

# log变换daysOnMarket
# train_data['daysOnMarket'] = np.log1p(train_data['daysOnMarket'])
# 对于原始的预测数据不需要进行log变换;只需要再程序末尾把预测好的数据进行反变换


# 4：对分类型数据进行label encode
def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1

            # 是否进行one_hot编码
            data[column] = data[column].astype('str')
    return data


# 对分类型数据进行one_hot编码:
def one_hot_encode(train_data,test_data):
    merge_data = pd.concat((train_data,test_data),axis=0)


    merge_data['buildingTypeId'] = merge_data['buildingTypeId'].astype('str')
    merge_data['tradeTypeId'] = merge_data['tradeTypeId'].astype('str')
    merge_data = pd.get_dummies(merge_data)
    train_data = merge_data[:train_data.shape[0]]
    test_data = merge_data[train_data.shape[0]:]
    print(train_data.head())
    print(test_data.head())
    return train_data,test_data








train_data = label_encode(train_data)
test_data = label_encode(test_data)
train_data,test_data = one_hot_encode(train_data,test_data)
print(train_data.head(100))
print(test_data.head(100))


# ---->>拆分数据,分为feature 和label 数据;
train = train_data.drop(columns='daysOnMarket')
test = test_data.drop(columns='daysOnMarket')


train_label = train_data['daysOnMarket']

test_label = test_data['daysOnMarket']


# 寻找超参数

# 参数的调节方法:



params = {
          # 'n_estimators': [100,300,500,1000,5000],# 300
          'max_depth':[x for x in range(5,6,1)],#5
          # 'max_depth':[x for x in range(3,10,1)],#5
          # 'learning_rate':[0.001,0.01,0.05,0.1,0.3,0.5,0.7,0.9], # 0.3
          # 'reg_alpha':[1e-5,1e-2,0.1,1,100],#1
          # 'gamma':[x for x in range(0,10,1)],#0
          # 'min_child_weight':[x for x in range(1,10,1)],# 5
          #   'subsample':[x for x in np.arange(0,1,0.01)], # 0.65
          }
'''
   eta –> learning_rate
   lambda –> reg_lambda
   alpha –> reg_alpha
'''
kfold = KFold(n_splits=10)
grid = GridSearchCV(estimator=XGBRegressor(),param_grid=params,scoring='neg_mean_absolute_error',cv=kfold)

# 训练
grid.fit(train,train_label)
# print(len(grid.cv_results_.values()))
# print(help(grid.))
# 打印最好参数和最好的得分值
print('best_params',grid.best_params_)
print('best_scoring',grid.best_score_)
print('best:%f use:%r'%(grid.best_score_,grid.best_params_))
for params, mean_score, scores in grid.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
model = grid.best_estimator_
print(model)



# 预测
# preds = np.expm1(model.predict(test))
preds = model.predict(test)

preds_Series = pd.Series(preds)
preds_df = pd.DataFrame(preds_Series,columns=['predictions'])
merge_data = pd.concat((origin_data,preds_df),axis=1)
merge_data.to_csv('./merge_result_xgboost.csv',index=False)

print(preds_Series.describe())
print(test_label.describe())
print('error',mean_absolute_error(test_label,preds))
print('pred_mean',preds.mean())
print('true_mean',test_label.mean())



