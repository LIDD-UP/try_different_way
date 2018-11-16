# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: pandas_settings.py
@time: 2018/11/16
"""
import pandas as pd

class PandasSettings(object):
    def __init__(self,max_columns,display_width):
        self.max_columns = max_columns
        self.display_width = display_width

    def pandas_settings(self):
        # 设置最大列显示
        pd.set_option('max_columns', self.max_columns)
        # 设置显示宽度
        pd.set_option('display.width', self.display_width)