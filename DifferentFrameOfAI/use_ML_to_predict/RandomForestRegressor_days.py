#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: RandomForestRegressor_days.py
@time: 2018/7/31
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import math
from scipy.stats import skew
import collections
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

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



# df = pd.read_csv('./company_house_data/month_6_1_try.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')
# tdf = pd.read_csv('./company_house_data/test_data_6_1_try.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

#
df = data_train_456
tdf = data_test_6
Y_label = tdf['daysOnMarket']


daysOnMarket = df['daysOnMarket']
df.fillna(df.mean(), inplace=True)

tdf.fillna(tdf.mean(), inplace=True)

y = df['daysOnMarket']
df = df.drop('daysOnMarket', axis=1)
X = df
tX = tdf.drop('daysOnMarket',axis=1)

train_num = len(X)
dataset = pd.concat(objs=[X, tX], axis=0)

#log transform skewed numeric features:
numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

dataset[skewed_feats] = np.log1p(dataset[skewed_feats])

dataset_preprocessed = pd.get_dummies(dataset)
train_preprocessed = dataset_preprocessed[:train_num]
test_preprocessed = dataset_preprocessed[train_num:]

X_train, X_test, y_train, y_test = train_test_split(train_preprocessed, y, test_size=0.3, random_state=0)

rmse_est = {}
for est in range(360,550,20):
    model = RandomForestRegressor(n_estimators=est, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = np.array(model.predict(X_test))
    rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)
    print ("RMSE: {0} - est: {1}".format(str(rmse), est))
    rmse_est[rmse]= est

d = collections.OrderedDict(sorted(rmse_est.items()))
print ('generating file')
model = RandomForestRegressor(n_estimators=list(d.items())[0][1], n_jobs=-1)
model.fit(train_preprocessed, y)
y_test_pred = model.predict(test_preprocessed)
print(test_preprocessed.head())
submission = pd.DataFrame({"Id": test_preprocessed.index,"daysOnMarket": y_test_pred})
submission.loc[submission['daysOnMarket'] <= 0, 'daysOnMarket'] = 0
fileName = "submission.csv".format(rmse)
submission.to_csv(fileName, index=False)

print(mean_absolute_error(Y_label,y_test_pred))

plt.plot(y_test_pred[0:100],c='red',label='预测值')
plt.plot(Y_label[0:100],c='green',label='真是值')
plt.title("RandomForestRegressor_days pre and label distribute circumstance")

plt.legend()
plt.show()











# # 加载测试数据
# X_test = tdf.loc[:,'province':'bedrooms']
# Y_test= tdf.daysOnMarket
# X_test = X_test.dropna()
# X_test = pd.get_dummies(X_test)
# X_test['add'] = pd.Series([col for col in range(len(X_test))])
# 
# 
# 
# print(df.head())
# print(tdf.head())
# # 加载训练数据
# X_train = df.loc[:,'province':'bedrooms']
# Y_train = df.daysOnMarket
# 
# # print(X_train.head())
# # print(X_train.dtypes)
# X_train = X_train.dropna()
# X_train['buildingTypeId'] = X_train['buildingTypeId'].astype(str)
# X_train = pd.get_dummies(X_train)
# print(X_train.head())
# 
# model = RandomForestRegressor()
# model.fit(X_train,Y_train)
# 
# y = model.predict(X_test)
# print(y)


