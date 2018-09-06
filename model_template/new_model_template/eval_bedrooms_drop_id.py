# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: eval_bedrooms_drop_id.py
@time: 2018/9/5
"""
import pandas as pd

pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice

'''
一个个添加特征进行测试，追重要一点是搞清楚tensorflowDNNregressor step 和batchsize的关系；

'''

data = pd.read_csv('month6_new.csv')


# 处理bedrooms
def process_bedrooms(data):
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        if i != 'nan':
            list_month_process.append(eval(i))
        else:
            list_month_process.append(i)
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('float')
    return data


# 处理bedrooms(eval)
data = process_bedrooms(data)