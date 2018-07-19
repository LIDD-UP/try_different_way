#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_column.py
@time: 2018/7/19
"""

'''
需要处理的问题；
1:bedrooms需要用eval函数求出值转化成int类型（或者float类型的数）
2：buildingtype要处理成两种类型的：House，和Condo

House类型的：如下
Row / Townhouse ：6
Duplex ：12
Residential ：3
Manufactured Home/Mobile ：10
Fourplex ：14
Mobile Home ：17
Garden Home: 5
Modular : 7

house:
3,5,6,7,9,10,12,13,14,16,17,18,

price 按50一下，50-100，100-150，150-200，200-400，400以上：

tradetype 分为sale lease 两种；




'''
import pandas as pd
import numpy as np



# 处理bedrooms

'''
a = '3+2'
b = eval(a)
print(b)
data = pd.read_csv('data.csv',header=0)
print(data.head())
data = data.dropna()
list_month = list(data['bedrooms'].astype('str'))
# print(list_month)
# a = pd.Series(list_month)
# print(a)
list_month_process = []
for i in list_month:
    print(type(i))
    b = eval(i)
    print(b)
    # list_month_process.append(eval(i))

print(list_month_process)

# print(eval(data['bedrooms'][4]))
# print(eval(data['bedrooms'][4]))
'''

# 处理bedrooms

def process_bedrooms(data_path):
    data = pd.read_csv(data_path,header=0)
    data = data.dropna()
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        list_month_process.append(eval(i))
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('float')
    return data

# new_data = process_bedrooms('data.csv')
# print(type(new_data['bedrooms'][1]))

# def to_new_csv_file(data):
#     data.tocsv('./')


# 处理buildingtypeid

'''
data = pd.read_csv('data.csv',header=0)
data = data.dropna()


house_list = [3,5,6,7,9,10,12,13,14,16,17,18]

# data['buildingTypeId'] = data[data.buildingTypeId not in house_list]

list_id = []
print(len(data['buildingTypeId']))
for i in data['buildingTypeId']:
    if i in house_list:
        i = 1
    else:
        i = 2
    list_id.append(i)
data['buildingTypeId'] = pd.Series(list_id)
print(data['buildingTypeId'].head())
print(list_id)
print(len(list_id))
'''

def convert_buildingtypeid(data):
    # data = pd.read_csv(data_path, header=0)
    # data = data.dropna()
    house_list = [3, 5, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18]
    list_id = []
    for i in data['buildingTypeId']:
        if i in house_list:
            i = 1
        else:
            i = 2
        list_id.append(i)
    data['buildingTypeId'] = pd.Series(list_id)
    return data

# new_data = convert_buildingtypeid('in_a_month.csv')
# print(new_data['buildingTypeId'].head())
# print(new_data.head())


def divide_by_tradetypeid(data,filename1,filename2):
    # data= data.drop(columns='id')
    new_data_1 = data[data.tradeTypeId==1]
    new_data_2 = data[data.tradeTypeId==2]
    new_data_1.to_csv('./{}.csv'.format(filename1),index=False)
    new_data_2.to_csv('./{}.csv'.format(filename2),index=False)



data_path = 'test_data_2.csv'
new_data = process_bedrooms(data_path=data_path)
new_data = convert_buildingtypeid(new_data)
new_data = new_data.drop(columns='id')

file1 = 'test_data2_1'
file2 = 'test_data2_2'
divide_by_tradetypeid(new_data,file1,file2)







