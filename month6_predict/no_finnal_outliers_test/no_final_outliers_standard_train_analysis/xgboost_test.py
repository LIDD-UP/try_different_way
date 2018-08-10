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
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

# train_data = pd.read_csv('./final_process_train_6.csv')
# test_data = pd.read_csv('./final_process_test_6.csv')

train_data = pd.read_csv('./no_final_Outliers_standard_train.csv')
# test_data = pd.read_csv('./no_final_Outliers_no_Standard_test_longitude125.csv')
test_data = pd.read_csv('./no_final_Outliers_no_Standard_test_1.csv')

train_data = train_data.dropna()
test_data = test_data.dropna()

# 数据处理log变换
def data_processing(data):
    data['price'] = np.log1p(data['price'])
    data['daysOnMarket'] = np.log1p(data['daysOnMarket'])
    return data

train_data = data_processing(train_data)
test_data = data_processing(test_data)


train = train_data.drop(columns=['daysOnMarket','bedrooms'])
test = test_data.drop(columns=['daysOnMarket','bedrooms'])

# 将test数据中的longitude 和latitude标准化，这是由于训练数据已经标准化了；
test['longitude'] = StandardScaler().fit_transform(np.array(test['longitude']).reshape(-1,1))
test['latitude'] = StandardScaler().fit_transform(np.array(test['latitude']).reshape(-1,1))


# 将分类数据ont_hot编码：
def dummies_process(data):
    data['buildingTypeId'] = data['buildingTypeId'].astype(int)
    data['buildingTypeId'] = data['buildingTypeId'].astype(str)
    # data['bedrooms'] = data['bedrooms'].astype(int)
    # data['bedrooms'] = data['bedrooms'].astype(str)
    data = pd.get_dummies(data)
    return data

train = dummies_process(train)
test = dummies_process(test)



train_label = train_data['daysOnMarket']
test_label = np.expm1(test_data['daysOnMarket'])


print(train.head())
print(test.head())

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
# print(help(XGBRegressor))

# 寻找超参数
params = {
          'n_estimators': [100,300,500,1000,5000],# 300
          'max_depth':[x for x in range(5,6,1)],#5
          'learning_rate':[0.001,0.01,0.05,0.1,0.3,0.5,0.7,0.9], # 0.3
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
preds = np.expm1(model.predict(test))
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
no_final_Outliers_no_Standard_test_1
error 8.849571322587554
pred_mean 16.084362
true_mean 18.706166868198306
'''

'''
no_final_Outliers_no_Standard_test_longitude125
error 8.878799913575554
pred_mean 16.01454
true_mean 18.669501822600242
'''



