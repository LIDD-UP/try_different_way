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

# 处理bedrooms

a = '3+2'
b = eval(a)
print(b)


