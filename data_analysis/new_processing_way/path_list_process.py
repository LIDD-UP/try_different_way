#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: path_list_process.py
@time: 2018/7/17
"""
import re

with open('./path_list_first.text','r') as f:
    for i in f.readlines():
        # print(type(i))
        if re.match('.*delislingDate.*', i) != None:
            print(i)

