#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_dict_values.py
@time: 2018/7/5
"""

# import time
# timestr = "time2009-12-14"
# t = time.strptime(timestr, "time%Y-%m-%d")
# print(t)


import tensorflow as tf
import os
import pandas as pd
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO) #答应出日志记录观察到一条：INFO:tensorflow:Restoring parameters from ./models/dnnlregressor\model.ckpt-400

dirname = os.path.dirname(os.getcwd())
train_filename = '\\use_estimator_new\\house_info.csv'
test_filename = '\\use_estimator_new\\test_house_info.csv'
# 加载训练数据



# 这里header 是零就可以了，这样才是从第一行开始的；
data = pd.read_csv(dirname+train_filename,header=0,usecols=[0,1,2,4,5,6,8,10,11,12,14] ,names=['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice', 'listingDate', 'daysOnMarket'])  # header等于一表示跳过第一行；只有指定列明之后才能用data['province']的方式取数据
# print(data[['longitude']]) #他是一个二位的数组，需要有两个【】这个；
data = data.dropna(axis=0)
example = data[['province', 'city', 'address', 'longitude', 'latitude', 'price','buildingTypeName', 'tradeTypeName', 'expectedDealPrice','listingDate']]
print(example['price'].values)

#把example['listingDate']拆分为年月日三个列；
list_data = list(example['listingDate'])
print(list_data[1])
a = list_data[1].split('/')
print(a)

list_break_together = []
for index,data in enumerate(list_data):
        list_break = data.split('/')
        list_break_together.append(list_break)



print(list_break_together)

# a = np.ndarray(list_break_together)
# b = np.array(list_break_together)
b = pd.DataFrame(list_break_together,columns=['year','month','day'],dtype='float32')
# print('a',a)
# print('b',int(b['year'].values))

b2 = b.values.astype('float')

print('b',b.values)
# print('b2',b2.values)

























# for index,data in enumerate(example['listingDate']):
#     # if data == '':
#
#     print(index,data)
# print(list(example['listingDate']))

# for i in example['listingDate']:
#     print(i)


# a = list('asss')
# b = a.replace('s','a',3)
# print(b)

# list_break_together =list()
# b = list_break_together.append('a')
# print(list_break_together,b)
#这里b为None，所以这个方法不返回任何东西；