#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: matplotStteing_pandas_show_setting.py
@time: 2018/8/1
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

pd.set_option('max_columns',200)
pd.set_option('display.width',1000)

import numpy as np
np.set_printoptions(suppress=True)


plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率



pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)