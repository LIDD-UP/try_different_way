#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: month_456_processing.py
@time: 2018/8/6
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# import missingno as msn

data = pd.read_csv('./company_house_data/month_456_1.csv')
# data = pd.read_csv('./month_6_processing.csv')
# data = data.drop(columns='price')
print(data.head())
print(data['buildingTypeId'].value_counts())

print(data.describe())
print('-=----------------------------------')

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure(figsize=(20,10)) #建立图像
p = data.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
x = p['fliers'][7].get_xdata() # 'fliers'即为离群点的标签
print(x)

y = p['fliers'][7].get_ydata()
print(',,',y)
y.sort() #从小到大排序，该方法直接改变原对象
print(y.min())
print(y.max())
print(len(y))

#用annotate添加注释
#其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#以下参数都是经过调试的，需要具体问题具体调试。
for i in range(len(x)):
  if i > 0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
plt.show() #展示箱线图
