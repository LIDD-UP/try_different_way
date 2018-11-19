# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: SklearnAPI.py
@time: 2018/11/19
"""
# 一般的线性方法
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

# xgboost
from xgboost import XGBRegressor

# 随机森林
from sklearn.ensemble import RandomForestRegressor

# 交叉验证
from sklearn.model_selection import cross_val_score

# 这里可能会用到模型的融合



class SklearnAPI(object):

    def xgboost_prediction(self):
        pass

    def randomforest_prediction(self):
        pass
