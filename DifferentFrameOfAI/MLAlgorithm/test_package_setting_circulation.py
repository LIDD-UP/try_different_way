# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_package_setting_circulation.py
@time: 2018/11/16
"""
import os
import sys
import pandas as pd
print(sys.path)
# import

# import approot
# dir = approot.get_root()
# print(dir)
# from DisplayAndPlotSettings.pandas_settings import PandasSettings
# pandas_settings = PandasSettings(100,200)
# pandas_settings.pandas_settings()

# train = pd.read_csv('./kaggle_house_data/train.csv')
# test = pd.read_csv('./kaggle_house_data/test.csv')
#
# print(train.head())
# print(test.head())
# print(train.shape)
import pandas as pd

from GetRootPath.approot import get_root
path = get_root()
print(path)
data = pd.read_csv(path+ '/DataFile/ML_data/kaggle_price_predict_data/train.csv')
print(data.head())



