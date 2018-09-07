# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: test_operation_py_file.py
@time: 2018/9/6
"""
base = '../in_different_days_to_predict/'
import os

def batch_execution_py_file():
    for i in range(3):
        if i ==0:
            base_path = 'in_{}_days'.format(i*7)
            py_name = 'in_{}_days_dnn.py'.format(i*7)
            os.chdir(base+base_path)
            os.system('python {}'.format(py_name))
        else:
            base_path = '../in_{}_days'.format(i * 7)
            py_name = 'in_{}_days_dnn.py'.format(i * 7)
            os.chdir(base_path)
            os.system('python {}'.format(py_name))

batch_execution_py_file()

