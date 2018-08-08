#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_outlier_of_coordinate_data.py
@time: 2018/8/8
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [1,2,3,4,5,66,77,32,57,23]
y = [1,2,3,4,5,77,777,333,555,12]

# 作图
plt.scatter(x,y)
plt.show()


def get_outlier(x,y,init_point_count ,distance,least_point_count):
    for i in range(len(x)):
        for j in range(len(x)):
             d =np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
             # print('距离',d)
             if d <= distance:
                init_point_count +=1
        if init_point_count <least_point_count+1:
            print(x[i],y[i])
        init_point_count =0


get_outlier(x,y,0,10,1)
