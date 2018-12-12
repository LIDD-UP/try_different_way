# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_quantile_based_buckets.py
@time: 2018/9/26
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.column', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


print(train_data['city'].value_counts())

sns.pairplot(train_data)
plt.tight_layout()
plt.show()
