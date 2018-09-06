#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_analysis.py
@time: 2018/8/10
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('no_final_Outliers_standard_train.csv')

print(data.describe())

sns.pairplot(data)
plt.show()