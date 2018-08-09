#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: oversample_2_test.py
@time: 2018/8/9
"""
import pandas as pd
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的分类算法SVC
from imblearn.ensemble import EasyEnsemble # 简单集成方法EasyEnsemble

# 导入数据文件
df = pd.read_table('data2.txt', sep=' ', names=['col1', 'col2','col3', 'col4', 'col5', 'label']) # 读取数据文件
x = df.iloc[:, :-1] # 切片，得到输入x
y = df.iloc[:, -1] # 切片，得到标签y
groupby_data_orgianl = df.groupby('label').count() # 对label做分类汇总
print (groupby_data_orgianl) # 打印输出原始数据集样本分类分布

# 使用SMOTE方法进行过抽样处理
model_smote = SMOTE() # 建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y) # 输入数据并作过抽样处理
x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=['col1','col2', 'col3', 'col4', 'col5']) # 将数据转换为数据框并命名列名
y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['label']) # 将数据转换为数据框并命名列名
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('label').count() # 对label做分类汇总
print (groupby_data_smote) # 打印输出经过SMOTE处理后的数据集样本分类分布

# 使用RandomUnderSampler方法进行欠抽样处理
model_RandomUnderSampler = RandomUnderSampler() # 建立RandomUnderSampler模型对象
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =model_RandomUnderSampler.fit_sample(x,y) # 输入数据并作欠抽样处理
x_RandomUnderSampler_resampled =pd.DataFrame(x_RandomUnderSampler_resampled,columns=['col1','col2','col3','col4','col5'])
# 将数据转换为数据框并命名列名
y_RandomUnderSampler_resampled =pd.DataFrame(y_RandomUnderSampler_resampled,columns=['label']) # 将数据转换为数据框并命名列名
RandomUnderSampler_resampled =pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled], axis= 1) # 按列合并数据框
groupby_data_RandomUnderSampler =RandomUnderSampler_resampled.groupby('label').count() # 对label做分类汇总
print (groupby_data_RandomUnderSampler) # 打印输出经过RandomUnderSampler处理后的数据集样本分类分布

