# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: pdn.py
@time: 2018/11/28
"""
from auto_ml import load_ml_model
from keras.models import load_model
import os
from my_conf.model_file_settings import model_auto_ml_path,model_keras_path

class MyPrediction(object):

    def my_predict_auto_ml(self,data):
        model = load_ml_model(model_auto_ml_path)
        result = model.predict(data)
        return result

    def my_predict_keras(self,data):
        model = load_model(model_keras_path)
        result = model.predict(data)
        return result