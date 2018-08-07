#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: use_processing_data_to predict.py
@time: 2018/8/7
"""
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

data_train_6 = pd.read_csv('./company_house_data/month_6_processing.csv')
data_test_6 = pd.read_csv('./company_house_data/test_data_6_1.csv')

