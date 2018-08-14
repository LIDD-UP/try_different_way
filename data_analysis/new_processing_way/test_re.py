#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_re.py
@time: 2018/7/17
"""
import re
a = 'realtor_history->Individual[2]->Websites[1]->Website'
# if re.match('.*days.*',a) !=None:
#     print(a)


# if re.findall('.*Individual\[[2]\].*', a) != None:
b  = re.findall('.*(Individual\[2]).*', a)
print(b)