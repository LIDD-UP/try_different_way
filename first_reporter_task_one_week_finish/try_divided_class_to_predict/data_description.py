# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_description.py
@time: 2018/9/26
"""
import pandas as pd


train_data = pd.read_csv('./input/test_inverse.csv')
test_data = pd.read_csv('./input/train_inverse.csv')

print('train:',train_data.shape)
print('test shape',test_data.shape)

train = train_data
test = test_data

train = train[train.month!=7]
print(train.shape)
train_2 = train_data[~(train_data.index.isin(train.index))]
print(train_2.shape)
print(train_2[train_2.month!=7])