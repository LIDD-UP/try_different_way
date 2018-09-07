# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: generage_directory.py
@time: 2018/9/6
"""
import os, sys
def genDir():
    base = '../in_different_days_to_predict/'
    i = 1
    for j in range(24):
        file_name = base + 'in_{}_days'.format(j*7)
        os.mkdir(file_name)
        i=i+1

genDir()