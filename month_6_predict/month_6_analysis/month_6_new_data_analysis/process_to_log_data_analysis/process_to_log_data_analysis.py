# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_to_log_data_analysis.py
@time: 2018/8/30
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice
import  missingno as msno

