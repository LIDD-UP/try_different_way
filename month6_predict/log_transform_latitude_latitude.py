#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: log_transform_latitude_latitude.py
@time: 2018/8/7
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


train_data = pd.read_csv('./final_process_train_6_dnn.csv')
test_data = pd.read_csv('./final_process_test_6_dnn.csv')
