#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_NaN_graph.py
@time: 2018/7/11
"""

import missingno
import pandas as pd
import os
import matplotlib.pyplot as plt

current_path = os.getcwd()
fil_name = '/dataset/{}.csv'.format('realtor_data')
file_path = current_path + fil_name
data = pd.read_csv(file_path)

plt.subplots(figsize=(100,100))

print(data.describe())
missingno.bar(data)
plt.show()
