#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: standard_longitude_latitude.py
@time: 2018/8/7
"""
'''
标准化longitude，latitude

'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv('./final_process_train_6_dnn.csv')
test_data = pd.read_csv('./final_process_test_6_dnn.csv')


train_data['longitude'] = StandardScaler().fit_transform(np.array(train_data['longitude']).reshape(-1,1))
train_data['latitude'] = StandardScaler().fit_transform(np.array(train_data['latitude']).reshape(-1,1))
test_data['longitude'] = StandardScaler().fit_transform(np.array(test_data['longitude']).reshape(-1,1))
test_data['latitude'] = StandardScaler().fit_transform(np.array(test_data['latitude']).reshape(-1,1))

# train_data = train_data[train_data.price>10]
# test_data = test_data[test_data.price>10]

# print(train_data.head())
# print(test_data.head())


# train_data =pd.get_dummies(train_data)
# test_data = pd.get_dummies(test_data)
#
# train_data.to_csv('./standard_longitude_latitude/standard_log_lat_train.csv',index=False)
# test_data.to_csv('./standard_longitude_latitude/standard_log_lat_test.csv',index=False)



# sns.pairplot(train_data)




# plt.figure(figsize=(20,10)) #建立图像
# p = train_data.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
# x = p['fliers'][1].get_xdata() # 'fliers'即为离群点的标签
# print(x)
# y = p['fliers'][5].get_ydata().max()
# print(y)

'''
latitude:3.4782735483824054
price:11.461642696843064
daysOnMarket:1.0986122886681098
'''
# sns.boxplot()
# p = plt.scatter(train_data['longitude'],train_data['latitude'])
# p.get_data()
# print(p)
# plt.show()


'''
需求：坐标空间中有无数个点：（平面坐标）
1:给定一个距离（d)，可以理解为半径
2:以平面内任意点为圆心，
3：定长得距离为半径得点的个数如果小于指定个数就输出该点的坐标；

'''

# def get_outlier(x,y,init_point_count ,distance,least_point_count):
#     for i in range(len(x)):
#         for j in range(len(x)):
#              d =np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
#              # print('距离',d)
#              if d <= distance:
#                 init_point_count +=1
#         if init_point_count <least_point_count+1:
#             print(x[i],y[i])
#         init_point_count =0
#
# get_outlier(test_data['longitude'],test_data['latitude'],0,0.3,1)


# data = pd.read_csv('./final_process_train_6.csv')
# data['price'] = np.log(data['price'])
#
# sns.pairplot(data)
# plt.show()


