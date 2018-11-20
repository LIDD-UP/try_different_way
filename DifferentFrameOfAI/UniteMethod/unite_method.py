# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: unite_method.py
@time: 2018/11/20
"""
# 统一以下训练的整个过程
# 1：包的导入和设置如pandas的控制台显示
import pandas as pd
import matplotlib.pyplot as plt
from DisplayAndPlotSettings.pandas_settings import PandasSettings
pandas_settings = PandasSettings(100,200)
pandas_settings.pandas_settings()

from xgboost import XGBRegressor

# 获取root目录
from GetRootPath.approot import get_root
path = get_root()
print(path)


# 2：定义类
class UniteMethod(object):
    # 1：构造函数
    def __init__(self):
        pass

    # 2：训练和预测数据的导入（此处的数据不能够合并，预测数据不能进行处理，只能进行特征的选择），
    # 同时可以把训练数据进行预测，去除掉一些差距比较大的数据）比较训练数据不同维度的分布情况
    def import_data(self,train_data_path,prediction_data_path):
        train_data = pd.read_csv(train_data_path)
        prediction_data = pd.read_csv(prediction_data_path)
        return train_data,prediction_data

    # 3：数据的预处理部分
    def preprocess_data(self,data,columns):
        data = data[columns]

        return data

    # 1：图形展示数据分布函数
    def plot_show(self,data):
        data.pairplot()
        plt.tight_layout()
        plt.show()

    # 4：训练函数(这里可能需要将不同的框架分隔成一个新的类，因为不同的框架可能导致前面数据处理方式不同）
    def train_function(self,train_data):
        xgb = XGBRegressor()
        xgb.fit(train_data)



# 1：auto_ml
# 2: xgbooot
# 3: randomforest
# 4: bagging
# 5: ridge
# 6: lasso
# 7: keras
# 8: tensorflow
# 5：预测函数
# 6: 结果保存函数
# 7：结果导入进行预测（应该是使用pickle）
# 3：main函数的运行
# 1：数据导入
# 2：数据处理
# 3：训练
# 4：预测



if __name__ == '__main__':
    data = pd.read_csv(path + '/DataFile/ML_data/kaggle_price_predict_data/train.csv')

    train = pd.read_csv(path + '/DataFile/ML_data/kaggle_price_predict_data/train.csv')
    test = pd.read_csv(path + '/DataFile/ML_data/kaggle_price_predict_data/test.csv')
