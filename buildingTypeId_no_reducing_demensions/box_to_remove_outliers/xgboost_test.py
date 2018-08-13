#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: xgboost_test.py
@time: 2018/8/10
"""
import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

# train_data = pd.read_csv('./final_process_train_6.csv')
# test_data = pd.read_csv('./final_process_test_6.csv')

train_data = pd.read_csv('./month_6_train_1.csv')
test_data = pd.read_csv('./test_data_1.csv')

# log 变换
train_data['price'] = np.log1p(train_data['price'])
test_data['price'] = np.log1p(test_data['price'])

# train_data['daysOnMarket'] = np.log1p(train_data['daysOnMarket'])
# test_data['daysOnMarket'] = np.log1p(test_data['daysOnMarket'])


# # 标准化
# train_data['longitude'] = StandardScaler().fit_transform(np.array(train_data['longitude']).reshape(-1,1))
# train_data['latitude'] = StandardScaler().fit_transform(np.array(train_data['latitude']).reshape(-1,1))
# test_data['longitude'] = StandardScaler().fit_transform(np.array(test_data['longitude']).reshape(-1,1))
# test_data['latitude'] = StandardScaler().fit_transform(np.array(test_data['latitude']).reshape(-1,1))

# 再将buildingTypeId 和 bedrooms dummies
# train_data['buildingTypeId'] = train_data['buildingTypeId'].astype(str)
# train_data['bedrooms'] = train_data['bedrooms'].astype(str)
#
# test_data['buildingTypeId'] = test_data['buildingTypeId'].astype(str)
# test_data['bedrooms'] = test_data['bedrooms'].astype(str)
#
# train_data = pd.get_dummies(train_data)
# test_data = pd.get_dummies(test_data)





train = train_data.drop(columns='daysOnMarket')
test = test_data.drop(columns='daysOnMarket')


train_label = train_data['daysOnMarket']
# test_label = np.expm1(test_data['daysOnMarket'])
test_label = test_data['daysOnMarket']


print(train.head())
print(test.head())

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
# print(help(XGBRegressor))

# 寻找超参数
params = {
          # 'n_estimators': [100,300,500,1000,5000],# 300
          'max_depth':[x for x in range(5,6,1)],#5
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
# preds = np.expm1(model.predict(test))
preds = model.predict(test)

preds_Series = pd.Series(preds)
print(preds_Series.describe())
print('error',mean_absolute_error(test_label,preds))
print('pred_mean',preds.mean())
print('true_mean',test_label.mean())

# 画图
plt.figure(figsize=(100,100))

plt.plot(preds[0:100],c='blue',label='pred')
plt.plot(test_label,c='red',label='true')
plt.title("RandomForest preds and true daysOnMarket distribution circumstance")
plt.legend()
plt.show()

'''
不对daysOnmarket进行log变化
count    673.000000
mean      17.084833
std        2.795359
min        9.554527
25%       15.398609
50%       17.030088
75%       18.995413
max       25.446043
dtype: float64
error 7.34644006301177
pred_mean 17.084833
true_mean 17.057949479940564

'''

'''
对longitude latitude 进行标准化：
count    673.000000
mean      18.151232
std        2.878635
min       10.566658
25%       16.929758
50%       18.293959
75%       20.152617
max       26.999578
dtype: float64
error 7.762931422705459
pred_mean 18.151232
true_mean 17.057949479940564

'''



