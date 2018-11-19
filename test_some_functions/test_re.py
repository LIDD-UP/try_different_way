# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_re.py
@time: 2018/11/19
"""
import re

source_str = r'abcdefg234?>\,.'
x = re.sub('[^A-Za-z]',' ',source_str)
print(x)