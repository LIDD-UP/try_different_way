# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: xgboost_to_predict.py
@time: 2018/8/23
"""

import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

data = pd.read_csv('./second.csv')
data = data[["longitude","latitude","price","buildingTypeId","bedrooms",'daysOnMarket']]

# predict_data = pd.read_csv('./predict_data.csv'')
# predict_data = predict_data[["longitude","latitude","price","buildingTypeId","bedrooms",'daysOnMarket']]


# data = data[[ 'longitude', 'latitude', 'price','daysOnMarket']]
# columns = [column for column in data.columns if data[column].dtype !='object']
# data = data.drop(columns='tradeTypeId')
# data = data[columns]
def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            # data[column] = data[column].astype('str')
    return data
data = label_encode(data)
data = data.drop(columns=['province','city','postalCode',])

# predict_data



data = data.dropna()

train_data ,test_data  = train_test_split(data,test_size=0.1)


print(train_data.shape)
print(test_data.shape)


# log 变换
# train_data['price'] = np.log1p(train_data['price'])
# test_data['price'] = np.log1p(test_data['price'])

# train_data['daysOnMarket'] = np.log1p(train_data['daysOnMarket'])
# test_data['daysOnMarket'] = np.log1p(test_data['daysOnMarket'])


train = train_data.drop(columns='daysOnMarket')
test = test_data.drop(columns='daysOnMarket')


train_label = train_data['daysOnMarket']
test_label = test_data['daysOnMarket']
# test_label = np.expm1(test_data['daysOnMarket'])


print(train.head())
print(test.head())


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV,KFold
# print(help(XGBRegressor))

# 寻找超参数
params = {
          # 'n_estimators': [100,300,500,1000,5000],# 300
          'max_depth':[x for x in range(3,4,1)],#5
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

grid = GridSearchCV(estimator=XGBRegressor(),param_grid=params,scoring='neg_mean_absolute_error')

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
preds = model.predict(test)
# preds = np.expm1(model.predict(test))

preds_Series = pd.Series(preds)
print(preds_Series.describe())
print('error',mean_absolute_error(test_label,preds))
print('pred_mean',preds.mean())
print('true_mean',test_label.mean())

# 画图
# plt.figure(figsize=(100,100))
#
# plt.plot(preds[0:100],c='blue',label='pred')
# plt.plot(test_label,c='red',label='true')
# plt.title("RandomForest preds and true daysOnMarket distribution circumstance")
# plt.legend()
# plt.show()

'''

'''

'''

'''



