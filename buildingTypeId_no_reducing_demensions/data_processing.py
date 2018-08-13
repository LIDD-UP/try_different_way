#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_processing.py
@time: 2018/8/13
"""
'''
原始数据的bedrooms算出，将tradeType的两种类型分开估计、
不对buildingTypeId 进行降维处理也就是只分成两类；
'''
import pandas as pd
import os


#  将bedrooms个数算出
def process_bedrooms(data_path):
    data = pd.read_csv(data_path,header=0)
    data = data.dropna()
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        list_month_process.append(eval(i))
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('str')
    return data


def divide_by_tradetypeid(data,filename1,filename2):
    # data= data.drop(columns='id')
    new_data_1 = data[data.tradeTypeId==1]
    new_data_2 = data[data.tradeTypeId==2]
    new_data_1.to_csv('./{}.csv'.format(filename1),index=False)
    new_data_2.to_csv('./{}.csv'.format(filename2),index=False)


# 获取month6数据的路径以及test_month6的数据路径
month_6_data_path = os.path.dirname(os.getcwd()) + '/input/raw_data/month_6.csv'
test_data_6_path = os.path.dirname(os.getcwd()) + '/input/raw_data/test_data_6.csv'


# month_6_data = pd.read_csv(month_6_data_path)
# test_data_6 = pd.read_csv(test_data_6_path)

month_6_data = process_bedrooms(month_6_data_path)
divide_by_tradetypeid(month_6_data, './input/month_6_1', './input/month_6_2')

test_data_6 = process_bedrooms(test_data_6_path)
divide_by_tradetypeid(test_data_6, './input/test_6_1', './input/test_6_2.')
