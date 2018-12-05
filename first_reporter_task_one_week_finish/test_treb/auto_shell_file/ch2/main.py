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
from my_conf.psql_settings import USER,PASSWORD,DBNAME
print('merge_data_path:',merge_data_path)
from keras.models import load_model
from my_conf.model_file_settings import model_keras_path
from tools.comput_ratio import compute_ratio
from sqlalchemy import create_engine
def transform_data_to_dataframe(data,column):
    data = pd.DataFrame(data,columns=[column])
    return data


def get_data_from_database(treb_sql_string):
    psql_tools = PSQLToos()
    conn = psql_tools.get_psql_connection_obj()
    sql_query_string = treb_sql_string
    prediction_data = pd.read_sql(sql_query_string,con=conn)
    return prediction_data

def save_data_to_database(prediction_result):
    engine = create_engine('postgresql://{0}:{1}@localhost:5432/{2}'.format(USER,PASSWORD,DBNAME))
    # create_engine说明：dialect[+driver]://user:password@host/dbname[?key=value..]
    # psql_tools = PSQLToos('test2')
    # conn = psql_tools.get_psql_connection_obj()
    prediction_result.to_sql(name='test',con=engine,if_exists='append')



def main():
    predict_or_test = "test"
    print(sys.argv[1])

    # 读取数据库数据
    # psql_tools = PSQLToos()
    # conn = psql_tools.get_psql_connection_obj()
    # sql_query_string = treb_sql_string
    # prediction_data = pd.read_sql(sql_query_string,con=conn)

    # 本地读取数据
    prediction_data = pd.read_csv('treb_toronto_11.csv')

    # 数据处理
    print('prediction data shape', prediction_data.shape)
    # auto_ml只需要处理一次，keras需要处理两次
    dp = DataProcess()
    prediction_data_after_process = dp.data_process(prediction_data, predict_or_test)
    # 然后保存与处理过后的文件
    orgin_data = prediction_data_after_process.reset_index(drop=True)
    # 去掉daysOnMarket
    prediction_data_after_process = prediction_data_after_process.drop(columns='daysOnMarket')
    print('prediction_data_after_process shape:',prediction_data_after_process.shape)
    my_prediciton = MyPrediction()
    # auto_ml 预测
    if sys.argv[1] == 'auto_ml':
        result_auto_ml = my_prediciton.my_predict_auto_ml(prediction_data_after_process)
        result_auto_ml_df = transform_data_to_dataframe(result_auto_ml, 'predictions')
        print(result_auto_ml_df.head())
        result_auto_ml_df.to_csv('./auto_ml_result.csv')

        save_data_to_database(result_auto_ml_df)
        merge_prediction_orgin_auto_ml = pd.concat((orgin_data, result_auto_ml_df), axis=1)
        compute_ratio(merge_prediction_orgin_auto_ml, 'predictions')
    # keras 预测
    if sys.argv[1] == 'keras':
        keras_process = dp.keras_data_process(prediction_data_after_process)
        result_keras = my_prediciton.my_predict_keras(keras_process)
        # 转化成DataFrame格式
        result_keras_df = transform_data_to_dataframe(result_keras, 'predictions')
        merge_prediction_origin_keras = pd.concat((orgin_data, result_keras_df), axis=1)
        compute_ratio(merge_prediction_origin_keras, 'predictions')
        save_data_to_database(result_keras_df)



if __name__ == '__main__':
    main()
    









