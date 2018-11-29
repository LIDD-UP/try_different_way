# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: model_file_settings.py
@time: 2018/11/28
"""
import os
import sys

auto_ml_model_name = 'model_auto_ml_9.h5'
keras_model_name = 'model_keras_test.h5'


model_auto_ml_path= os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/model_file/{}'.format(auto_ml_model_name)
print(model_auto_ml_path)
model_keras_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/model_file/{}'.format(keras_model_name)
print(model_keras_path)
