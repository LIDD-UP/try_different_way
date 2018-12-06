# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: main.py
@time: 2018/11/27
"""
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sqlalchemy import create_engine

from tools.psql_tools import PSQLToos
from new_pdn.pdn import MyPrediction
from sql_script.sql_script import treb_sql_string,prediciton_query_string
from data_process.data_process import DataProcess
from my_conf.merge_data_file_for_dummies_settings import merge_data_path
from my_conf import psql_settings as jdbc
from my_conf.model_file_settings import model_keras_path
from tools.comput_ratio import compute_ratio
from tools.send_email import send_email



def transform_data_to_dataframe(data,column):
    data = pd.DataFrame(data,columns=[column])
    return data


def get_data_from_database(treb_sql_string):
    psql_tools = PSQLToos()
    conn = psql_tools.get_psql_connection_obj()
    sql_query_string = treb_sql_string
    prediction_data = pd.read_sql(sql_query_string,con=conn)
    return prediction_data

def save_data_to_database(prediction_result,origin_data,table_name,columns_list):
    # local database connect
    engine = create_engine('postgresql://{0}:{1}@localhost:5432/{2}'.format(jdbc.USER,jdbc.PASSWORD,jdbc.DBNAME))
    # create_engine说明：dialect[+driver]://user:password@host/dbname[?key=value..]
    columns_list = [
        'estateMasterId',
        'realtorDataId',
        'realtorHistoryId',
        'mlsNumber',
    ]
    origin_data_need = origin_data[columns_list]
    merge_data = pd.concat((origin_data_need,prediction_result),axis=1)
    merge_data.to_sql(name=table_name,con=engine,if_exists='append')



def main():
    # try:
    predict_or_test = "test"
    print(sys.argv[1])

    # 读取数据库数据
    psql_tools = PSQLToos()
    prediction_data = psql_tools.get_data(True)

    # 本地读取数据
    # prediction_data = pd.read_csv('treb_toronto_11.csv')


    # 数据处理
    print('prediction data shape', prediction_data.shape)
    total_data_number = prediction_data.shape[0]
    # auto_ml只需要处理一次，keras需要处理两次
    dp = DataProcess()
    prediction_data_after_process = dp.data_process(prediction_data, predict_or_test)
    # 然后保存与处理过后的文件
    origin_data = prediction_data_after_process.reset_index(drop=True)
    # 去掉daysOnMarket 和不用的预测特征
    prediction_data_after_process = prediction_data_after_process.drop(columns=[
        'daysOnMarket',
        'estateMasterId',
        'realtorDataId',
        'realtorHistoryId',
        'mlsNumber'
    ])
    print('prediction_data_after_process shape:',prediction_data_after_process.shape)
    prediction_data_number = prediction_data_after_process.shape[0]
    my_prediciton = MyPrediction()
    # auto_ml 预测
    if sys.argv[1] == 'auto_ml':
        result_auto_ml = my_prediciton.my_predict_auto_ml(prediction_data_after_process)
        result_auto_ml_df = transform_data_to_dataframe(result_auto_ml, 'predictions')
        print(result_auto_ml_df.head())
        # result_auto_ml_df.to_csv('./auto_ml_result.csv')
        # 保存到本地数据库
        save_data_to_database(result_auto_ml_df, origin_data, 'ch_test', [])
        # 保存到产品数据库
        # 。。。

        # 计算比例
        merge_prediction_orgin_auto_ml = pd.concat((origin_data, result_auto_ml_df), axis=1)
        # compute_ratio(merge_prediction_orgin_auto_ml, 'predictions')


    # keras 预测
    if sys.argv[1] == 'keras':
        keras_process = dp.keras_data_process(prediction_data_after_process)
        result_keras = my_prediciton.my_predict_keras(keras_process)
        # 转化成DataFrame格式
        result_keras_df = transform_data_to_dataframe(result_keras, 'predictions')
        # 保存到本地数据库：
        save_data_to_database(result_keras_df, origin_data,'ch_test', [])
        # 保存到产品数据库
        # 。。。。

        # 计算比例
        merge_prediction_origin_keras = pd.concat((origin_data, result_keras_df), axis=1)
        # compute_ratio(merge_prediction_origin_keras, 'predictions')


    send_email("AI预测", '总共查询:{0}条数据，使用{1}方法预测了:{2}'.format(total_data_number, sys.argv[1], prediction_data_number))
    # except Exception as e:
    #     print(e)
    #     send_email("AI预测", '预测失败：{}'.format(e))



if __name__ == '__main__':
    main()
    









