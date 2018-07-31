#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: ML_predictdays.py
@time: 2018/7/31
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.metrics import mean_absolute_error
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False


import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
pd.set_option('max_columns',200)
pd.set_option('display.width',1000)
train = pd.read_csv('./company_house_data/month_6_1_try.csv')
test = pd.read_csv('./company_house_data/test_data_6_1_try.csv')

# print(train.head())
# print(test.head())
# print(train.shape)
# print(test.shape)

all_data = pd.concat((train.loc[:,'province':'bedrooms'],
                      test.loc[:,'province':'bedrooms']))
# print(all_data.head())
# print(all_data.shape)
#
# print(all_data.loc[train.index].shape)
# print(all_data.loc[test.index].shape)

prices = pd.DataFrame({"days":train["daysOnMarket"], "log(days + 1)":np.log1p(train["daysOnMarket"])})

# print(prices.head())
# print(prices.shape)
# prices.hist()
# plt.show()
'''
(125636, 11)
(860, 11)
   days  log(days + 1)
0      8        2.197225
1      8        2.197225
2     32        3.496508
3     82        4.418841
4     56        4.043051
(125636, 2)

'''
train["daysOnMarket"]  = np.log1p(train["daysOnMarket"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# print(all_data.shape)
# print(all_data.head())
# print(all_data.head(1000))
# print(len(all_data))
all_data = pd.get_dummies(all_data,sparse=True)
print(all_data.head())



all_data = all_data.fillna(all_data.mean())
print('fillna done')

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.daysOnMarket

print('load data success')


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
model_ridge = Ridge()


print('岭回归完成')

print('作图')
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

print(cv_ridge.min())






# 利用LASSO
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print(rmse_cv(model_lasso).mean())

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# found most important coefficients are
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
print(preds.head())
true_pred_data = np.array(np.expm1(preds['preds']))
print('preds',true_pred_data)

true_label_data = np.array(np.expm1(preds['true']))
print('label',true_label_data)
print(mean_absolute_error(true_label_data,true_pred_data))
# 画图：
list_true_lable_data = list(true_label_data)
list_true_preds_data = list(true_pred_data)
plt.plot(list_true_lable_data,c='red',label='真实值')
plt.plot(list_true_preds_data,c='green',label='预测值')
plt.legend()
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()