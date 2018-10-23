# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/10/23
"""
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

data_train = pd.read_csv('./input/test_treb_month_78.csv')
data_train_all = pd.read_csv('./input/treb_all_column_month_3to8.csv')

data_test = pd.read_csv('./input/treb_test_month_9.csv')
data_test_all = pd.read_csv('./input/treb_test_all_column_month_9.csv')

msno.bar(data_test_all)
plt.tight_layout()
plt.show()

