# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: pdn.py
@time: 2018/11/28
"""
from auto_ml import load_ml_model
from keras.models import load_model
import os
from my_conf.model_file_settings import model_auto_ml_path

class MyPrediction(object):

    def my_predict_auto_ml(self,data):
        # data_path = os.path.dirname(os.path.abspath(os.path.curdir)) + '/ch/model_file/model_auto_ml_9.h5'
        model = load_ml_model(model_auto_ml_path)
        result = model.predict(data)
        return result

    def my_predict_keras(self,data):
        data_path = os.path.dirname(os.path.abspath(os.path.curdir)) + '/ch/model_file/model_keras_test_7_5_9.h5'
        model = load_ml_model(data_path)
        result = model.predict(data)
        return result