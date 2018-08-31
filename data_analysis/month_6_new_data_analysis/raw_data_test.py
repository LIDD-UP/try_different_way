# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: raw_data_test.py
@time: 2018/8/30
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice


data = pd.read_csv('./month6_new.csv')




# print(data['garageSpaces'].mode())
# print(data['lotFront'].mode())
# print(data['style'].mode())
#
# data['style'] = data['style'].fillna(str(data['style'].mode()[0]))
# print(data['style'])

'''
利用调试工具一步步的画图分析数据，去除异常值：
'''
data = data.drop(columns=['id','propertytypeid','elevator',
        'longitude', # 这个分布不符合正太分布，比较麻烦，长尾分布；
        'latitude', # 同longitude
        'price',
        ])

def remove_numeric_flier_data(data):
    data = data[data.tradeTypeId==1]
    data['buildingTypeId'] = data['buildingTypeId'].astype('str')
    # print(len(data[data.longitude>-60]))
    # data = data[data.longitude<-60]
    '''
    # 由于观察longitude的直方图发现longitude是长尾分布；有很大一部分数据都在-95到-85之间；
    对于长尾分布的数据的处理方式是截尾或者是缩尾截尾就是把尾部数据去除，
    缩尾就是将尾部数据合并成一个数据（用平均值或者是众数代替，分位数代替；平均值或者众数不太合适）
    ：也就是可以看成一种分箱操作，但是分箱其实是将数据转化成了类别型的数据；
    而前面的处理还可以将他看成连续型的数据；
    还可以使用非参数检验的方法：
    '''
    # print(len(data[data.latitude <=42])) # 50
    # print(len(data[data.latitude>=57])) # 75
    # data = data[data.latitude>42]
    # data = data[data.latitude<57]
    '''
    latitude 也是一个长尾分布的数据；跟longitude一样：数据的根据地理位置信息；
    应该大于43，但是由于舍弃的数据量有2000多条，暂定把界限定在42，顶点定在：57；
    这个数据呈现起伏的情况，跟longitude一样中间一部分数据很少，
    考虑要不要根据经纬度的情况，将模型分成两个部分；
    
    '''
    # print(len(data[data.price>2000000]))
    # data = data[data.price<2000000]
    # data['price'] = data['price'].apply(lambda x: x if x<100000 else 1500000)
    # print(data['price'])
    '''
    price 从严格意义来件才属于长尾分布，也就是说需要将price过于离谱的数据
    去除再取1%分位点数据值作为大于1%分位点数的值，也就是利用缩尾的方法；
    但是用这种方案又导致尾部有起伏的趋势，所以还是直接用截尾的方式来做了
    但是并没有用到1%这个分位点来做，只是用了直观得观察来确定得值；
    '''




    return data

data = remove_numeric_flier_data(data)



for column in data.columns:
    if column != 'daysOnMarket' and data[column].dtype != 'object':
        plt.scatter(data[column],data['daysOnMarket'])
        plt.xlabel(column)
        plt.ylabel('daysOnMarket')
        plt.show()
        plt.hist(data[column])
        plt.show()

print('finish')
# sns.pairplot(data)
# plt.show()


# 对数值数据进行处理：一个个特征进行处理，主要去除异常值了离群点：









