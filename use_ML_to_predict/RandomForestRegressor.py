#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: RandomForestRegressor.py
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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["dir", "./kaggle_house_data"]).decode("utf8"))


df = pd.read_csv('./kaggle_house_data/train.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')
tdf = pd.read_csv('./kaggle_house_data/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

salePrice = df['SalePrice']

'''
You can try different ranges of n_estimators for the RandomForrestRegressor. 
I am saving these values in an Ordered dictionary to get the n_estimator value for 
east RMSE value Python 2.7 code is here 
https://github.com/zubairdotnet/HousePriceRegression/blob/master/RFRWithSkewness.py
'''


df.fillna(df.mean(), inplace=True)
TotalBsmtSFMean = df['TotalBsmtSF'].mean()
df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

tdf.fillna(tdf.mean(), inplace=True)
TTotalBsmtSFMean = tdf['TotalBsmtSF'].mean()
tdf.loc[tdf['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)

y = df['SalePrice']
df = df.drop('SalePrice', axis=1)
X = df
tX = tdf

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
submission = pd.DataFrame({"Id": test_preprocessed["Id"],"SalePrice": y_test_pred})
submission.loc[submission['SalePrice'] <= 0, 'SalePrice'] = 0
fileName = "submission.csv".format(rmse)
submission.to_csv(fileName, index=False)