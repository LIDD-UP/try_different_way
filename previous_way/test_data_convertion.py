#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_data_convertion.py
@time: 2018/7/6
"""


# dict1 = {'a':'1','b':'2'}
# c = int(dict1)
# # a = ['1','3','4']
# # b = dict(a)
# print(int(c))
import pandas as pd
import numpy as np

a = [
    [1,2,3],
    [4,5,6],
]

b = pd.DataFrame(a,columns=['year','month','day'])
c = np.array(b)
print(b)
