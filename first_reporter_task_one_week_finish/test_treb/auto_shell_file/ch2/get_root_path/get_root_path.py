# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_root_path.py
@time: 2018/11/27
"""
import os
import sys


def get_root_path():
    path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(path)
