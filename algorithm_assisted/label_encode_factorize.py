# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: label_encode_factorize.py
@time: 2018/8/28
"""
def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            data[column] = data[column].astype('str')
    return data