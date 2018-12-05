# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: merge_data_file_for_dummies_settings.py
@time: 2018/11/29
"""
import os
file_name = 'treb_toronto_3to8_1' #treb_toronto_3to8_1,merge_data_for_dummies
merge_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/merge_data_for_dummies/{}.csv'.format(file_name)

print(merge_data_path)