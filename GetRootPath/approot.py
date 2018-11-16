# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: approot.py
@time: 2018/11/16
"""


import os

def get_root():
    return os.path.dirname(os.path.dirname( os.path.abspath( __file__ ) ))


if __name__ == '__main__':
    path = get_root()
    # print(os.path.abspath(__file__))
    # print(os.path.curdir)
    # print(os.getcwd())
    print(path)
    # os.chdir(r'F:\virtualenv\scrapy\Scripts')
    # print(os.getcwd())
    # print(os.system('scrapy genspider'))