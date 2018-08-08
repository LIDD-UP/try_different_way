#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: stdandard_price.py
@time: 2018/7/19
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# data= pd.read_csv('in_a_month.csv',header=0)

# data['price'] = StandardScaler.fit()

# max_value = np.sort(data['longitude'])
# data = data[data.longitude>0]
# print(data['longitude'])
# 发现精度有大于0的且值基本为79，判定为异常数据

# data_max= data['latitude'].max()  # 60.876148
# data_min = data['latitude'].min() # 4.436977
# data_exception = data[data.latitude<30]
'''
1537     34.770301
4403     32.843119
5176     40.250025
7376     38.601765
8451     39.963035
8974     38.164065
11410    25.917971
13269    38.714000
15512    37.808425
16101    29.870841
18126    29.980508
19478    37.328981
19898     4.436977
21463    40.888706
21647    39.138228
23137    36.212158
25534    36.059344
27216    35.950754
29097    35.950754
'''

# print(data_exception['latitude'])
#
'''
精度每0.001 为100米
维度每隔0，001 为111米；
0.005
0.001*(500/111)
'''
# 50,100

# print(50/0.005)

# min = 10
# max = 100

# for i in range(min,max,0.001):
#     print(i)
# np.sort()

list_len = (20 -10)/0.005
list_boundaries =[]
middle =10
for i in range(int(list_len)):
    middle += 0.005

    list_boundaries.append(middle)


print(sorted(list_boundaries))

