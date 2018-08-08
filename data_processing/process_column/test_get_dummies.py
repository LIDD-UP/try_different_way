#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_get_dummies.py
@time: 2018/7/31
"""
import pandas as pd
train = pd.read_csv('./company_house_data/month_456_1.csv')
test = pd.read_csv('./company_house_data/test_data_6_1.csv')

train_binary = pd.get_dummies(train)
print(train_binary)

