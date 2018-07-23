#-*- coding:utf-8 _*-
"""
@author:bluesli
@file: draw.py
@time: 2018/07/22
"""

import re
import matplotlib.pyplot as plt
column_list=[]
prediction_mean_list = []
label_mean_list = []
error_list = []

with open('./log.txt','r') as f:
     for i in f.readlines():
          b = i.split(',')
          column_list.append(b[0])
          prediction_mean_list.append(float(b[1]))
          label_mean_list.append(float(b[2]))
          error_list.append(float(b[3]))



print(len(column_list))
print(column_list)
print(prediction_mean_list)
print(label_mean_list)
print(error_list)


# plt.figure(figsize=(100,100))
plt.title('Result Analysis')
plt.xticks(rotation=45)
plt.plot(column_list, prediction_mean_list, color='green', label='pre_mean')
plt.plot(column_list, label_mean_list, color='blue', label='label_mean')
plt.plot(column_list, error_list,  color='red', label='error')

plt.legend() # 显示图例
plt.tick_params(labelsize=5)
plt.xlabel('Column')
plt.ylabel('Value')

plt.show()