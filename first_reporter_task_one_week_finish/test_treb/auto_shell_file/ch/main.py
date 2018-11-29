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

def transform_data_to_dataframe(data):
    data = pd.DataFrame(data)
    return data

if __name__ == '__main__':

    # 读取数据库数据
    psql_tools = PSQLToos()
    conn = psql_tools.get_psql_connection_obj()
    sql_query_string = treb_sql_string
    prediction_data = pd.read_sql(sql_query_string,con=conn)

    # prediction_data = pd.read_csv('treb_toronto_9.csv')
    print('prediction_data',prediction_data.shape)
    prediction_data_after_process = DataProcess.data_process(prediction_data)




    # print('prediction_data_after_process shape:',prediction_data_after_process.shape)
    dp = DataProcess()
    keras_process = dp.keras_data_process(prediction_data_after_process)
    # print('keras_process :shape:',keras_process.shape)
    my_prediciton = MyPrediction()
    result_auto_ml = my_prediciton.my_predict_auto_ml(prediction_data)
    result_keras = my_prediciton.my_predict_keras(keras_process)
    data1 = transform_data_to_dataframe(result_auto_ml)
    data2 = transform_data_to_dataframe(result_keras)
    print(data1.head())
    data1.to_csv('./auto_ml_result.csv')
    merge_data = pd.concat((data1,data2),axis=1)
    merge_data.to_csv('./merge_auto_keras.csv')
    print(data2.head())









