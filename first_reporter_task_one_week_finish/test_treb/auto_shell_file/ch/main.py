# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: main.py
@time: 2018/11/27
"""
import os,sys
path = os.path.abspath(__file__)
print(path)
sys.path.append(os.path.abspath(__file__))
print(path)
from tools.psql_tools import PSQLToos
from new_pdn.pdn import MyPrediction
# from tools.do_predictions import MyPrediction
import pandas as pd
from sql_script.sql_script import treb_sql_string
# from do_prediction import do_predictions
from data_process.data_process import DataProcess
from my_conf.merge_data_file_for_dummies_settings import merge_data_path
print('merge_data_path:',merge_data_path)
from keras.models import load_model
from my_conf.model_file_settings import model_keras_path
from tools.comput_ratio import compute_ratio

def transform_data_to_dataframe(data,column):
    data = pd.DataFrame(data,columns=[column])
    return data

if __name__ == '__main__':

    # 读取数据库数据
    # psql_tools = PSQLToos()
    # conn = psql_tools.get_psql_connection_obj()
    # sql_query_string = treb_sql_string
    # prediction_data = pd.read_sql(sql_query_string,con=conn)

    # 本地读取数据
    prediction_data = pd.read_csv('treb_toronto_11.csv')

    # 数据处理
    print('prediction_data',prediction_data.shape)
    # auto_ml只需要处理一次，keras需要处理两次
    prediction_data_after_process = DataProcess.data_process(prediction_data)
    # 然后保存与处理过后的文件
    orgin_data = prediction_data_after_process.reset_index(drop=True)
    # 去掉daysOnMarket
    prediction_data_after_process = prediction_data_after_process.drop(columns='daysOnMarket')
    # print('prediction_data_after_process shape:',prediction_data_after_process.shape)
    dp = DataProcess()
    keras_process = dp.keras_data_process(prediction_data_after_process)
    # print('keras_process :shape:',keras_process.shape)
    # 预测
    my_prediciton = MyPrediction()
    # auto_ml 预测
    result_auto_ml = my_prediciton.my_predict_auto_ml(prediction_data_after_process)
    # keras预测
    result_keras = my_prediciton.my_predict_keras(keras_process)
    # 转化成DataFrame格式
    result_auto_ml_df = transform_data_to_dataframe(result_auto_ml,'predictions')
    result_keras_df = transform_data_to_dataframe(result_keras,'predictions')
    # 测试
    print(result_auto_ml_df.head())
    result_auto_ml_df.to_csv('./auto_ml_result.csv')
    merge_data = pd.concat((result_auto_ml_df,result_keras_df),axis=1)
    merge_data.to_csv('./merge_auto_keras.csv')
    print(result_keras_df.head())

    # 将预测数据和原始数据合并计算比例：
    merge_prediction_orgin = pd.concat((orgin_data,result_auto_ml_df),axis=1)
    compute_ratio(merge_prediction_orgin,'predictions')
    # 读入数据库
    









