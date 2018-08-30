# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: raw_data_test.py
@time: 2018/8/30
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice


data = pd.read_csv('./month6_new.csv')



print(data['garageSpaces'].mode())
print(data['lotFront'].mode())
print(data['style'].mode())

data['style'] = data['style'].fillna(str(data['style'].mode()[0]))
print(data['style'])