#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: form_reporter_table.py
@time: 2018/7/11
"""
import os
import pandas as pd
import pandas_profiling


def form_reporter_table(filename):
    current_path = os.getcwd()
    fil_name = '/{}.csv'.format(filename)
    file_path = current_path + fil_name
    data = pd.read_csv(file_path)
    print(data)
    prf = pandas_profiling.ProfileReport(data)
    prf.to_file('./{}.html'.format(filename))


if __name__ == '__main__':
    form_reporter_table('realtor_data')
