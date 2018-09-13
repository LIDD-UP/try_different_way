# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: datanaalysis.py
@time: 2018/9/13
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
import  missingno as msno
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./input/train_month_7_d.csv')

msno.bar(data)
plt.tight_layout()
plt.show()