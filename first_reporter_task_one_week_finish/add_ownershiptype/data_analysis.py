# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/9/19
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


train_data = pd.read_csv('./input/month_7_delisting_after_process_2.csv')

# msno.bar(train_data)
# plt.show()

print(train_data['city'].describe())


