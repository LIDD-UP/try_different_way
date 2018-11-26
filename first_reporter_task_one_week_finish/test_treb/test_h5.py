# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_h5.py
@time: 2018/11/26
"""
import h5py

x = h5py.File('model.h5')
for i in x:
    print(i)

# print(x)
# print(x.keys())
# print(x.values())
print(type(x['model_weights']))
