# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: keras_test_1.py
@time: 2018/11/26
"""

# -*- coding:utf-8 _*-
""" 
@author:Administrator
@file: keras_auto_shell.py
@time: 2018/11/23
"""

import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

def preprocess_data(data):
    data = data[[
        "longitude",
        "latitude",
        # "city",
        # "province",
        "price",
        "tradeTypeId",
        # # "listingDate",
        "buildingTypeId",
        "bedrooms",
        "bathroomTotal",
        # 'postalCode',
        'daysOnMarket',
        'ownerShipType',
        # 'projectDaysOnMarket',
        'district',

        # 以下就是用于测试得新得特征；
        'style', #22.769283885157083
        'community', # 类似于city类型得数据，类型有766个； #22.38147912725983
        'airConditioning', #22.755048806968883
        'washrooms', # 连续 #23.691205780782205
        'basement1',# 地下室22.797430800725444
        'familyRoom', # 22.794731300998404
        'fireplaceStove', # 2 w 左右 #22.82878318024665
        'heatSource', # 数据量可以2w+# 22.75554140962404
        'garageType', # 2 w+ #22.79707321027956
        'kitchens', # 22.79393809434976
        'parkingSpaces', #22.807931672409705
        #
        'parkingIncluded',#22.786586056260784
        'rooms',# 22.785397232054713

        'waterIncluded', # 22.80653144493355
        'totalParkingSpaces', # 22.81551411353129
        #
        # 'frontingOn',  # 面向得方向，drop掉之后有1w多:14270
        # 'drive',  # 14270
        # 'pool',  # 这个偏少；14270
        # 'sewers',  # 这个数据比较少 1w+：14270

        # more column
        # 'room3',

]]
    # 是否只考虑tradeTypeId为1的数据
    # data = data[data.tradeTypeId == 1]
    # data = data.drop(columns=['tradeTypeId'])
    print(data.shape)
    data = data.dropna(axis=0)
    print(data.shape)
    bedrooms_list = []
    for bedrooms in data["bedrooms"]:
        # print(bedrooms)
        if isinstance(bedrooms,float):
            bedrooms_list.append(int(bedrooms))
        elif isinstance(bedrooms,int):
            bedrooms_list.append(int(bedrooms))
        else:
            bedrooms_list.append(int(eval(bedrooms)))
    data["bedrooms"] = bedrooms_list
    bathroom_total_list = []
    for bathroom_total in data["bathroomTotal"]:
        bathroom_total_list.append(int(bathroom_total))
    data["bathroomTotal"] = bathroom_total_list

    # 将price做log变换
    # data['price'] = np.log1p(data['price'])

    # 需要进行一次one_hot 编码；
    # data = pd.get_dummies(data)


    return data


def compute_ratio(data):
    data_10 = []
    data_20 = []
    data_30 = []
    data_more = []

    # data = data.drop(columns=['index'])

    # test_column = 'projectDaysOnMarket'
    test_column = 'predictions'

    for i in range(len(data)):
        # print(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 10:
            data_10.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 10 and abs(
                data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 20:
            data_20.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 20 and abs(
                data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) <= 30:
            data_30.append(i)
        if abs(data.iloc[i][test_column] - data.iloc[i]['daysOnMarket']) > 30:
            data_more.append(i)

    print(len(data_10) / len(data))
    print(len(data_20) / len(data))
    print(len(data_30) / len(data))
    print(len(data_more) / len(data))
    print(mean_absolute_error(data[test_column], data['daysOnMarket']))
    return len(data_10) / len(data), len(data_20) / len(data), len(data_30) / len(data), len(data_more) / len(data),mean_absolute_error(data[test_column], data['daysOnMarket'])



def compute_ratio2(data):
    test_column = 'predictions'
    data_len = len(data)
    print(data[abs(data.predictions - data.daysOnMarket) <= 10]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 10 and abs(
                data.predictions - data.daysOnMarket) <= 20]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 20 and abs(
                data.predictions - data.daysOnMarket) <= 30]/data_len)
    print(data[abs(data.predictions - data.daysOnMarket) > 30]/data_len)
    print(mean_absolute_error(data[test_column], data['daysOnMarket']))

def first_train(train_data,train_data_label,epochs,f):
    model = Sequential()

    model.add(Dense(16, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(32, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(64, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(128, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(256, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(516, activation="relu", input_dim=X_train_data.shape[1]))
    model.add(Dense(1, input_dim=X_train_data.shape[1], kernel_regularizer=l1(0.1)))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(train_data, train_data_label, epochs=epochs, )
    model.save('model.h5')
    f.write('this is first train ,model is save as model.h5\n')
    return model


def continue_train(train_data,test_data,epochs,f,i):
    model = load_model('model.h5')
    model.fit(train_data,test_data,epochs=epochs)
    model.save('model.h5')
    f.write('this is another train and will remove train data %d'%i)
    return model


def test_data_save_and_merge_data(prediction_result,origin_data,f):
    x = prediction_result
    x_dataframe = pd.DataFrame(x, columns=['predictions'])
    merge_data = pd.concat((origin_data, x_dataframe), axis=1)
    merge_data_df = pd.DataFrame(merge_data)
    merge_data_df.to_csv('./merge_data_bak/merge_data_auto_ml.csv', index=False)
    print(x_dataframe.describe())
    print(df_test_label.describe())

    print(mean_absolute_error(df_test_label, x))
    x,y,z,h,g = compute_ratio(merge_data_df)
    f.write("""in this time ,we predict test data,and result as follows\n
            in 10:{0}\n
            10 to 20:{1}\n
            20 to 30:{2}\n
            more 30:{3} \n
            mean_absolute_error:{4}\n
            """.format(x,y,z,h,g))


def train_data_save_and_merge(model,origin_data_train,X_train_data,f):
    # 预测训练数据
    print(origin_data_train.head())
    print(origin_data_train.shape)
    train_prediction = model.predict(X_train_data)
    train_dataframe = pd.DataFrame(train_prediction, columns=['predictions'])
    merge_train_data = pd.concat((origin_data_train, train_dataframe), axis=1)
    merge_train_data_df = pd.DataFrame(merge_train_data)

    merge_train_data_df.to_csv('./merge_data_bak/merge_train_data.csv', index=False)
    f.write('finish train data prediction and save the merge the train data and label\n')



def process_train_merge_data_remove_some_data(data,f,remove_ratio):
    data_orgin = data

    # print(data.head())
    print(data.shape)
    data = data[abs(data.predictions - data.daysOnMarket) < remove_ratio]
    # print(data.head())
    print(data.shape)
    data.to_csv('./input/treb_toronto_3to8_1.csv')
    data_orgin.to_csv('./orgin_data.csv')
    f.write('finish the remove big gap of train data ,remove ratio is:{0}and train data shape is:{1}\n'.format(remove_ratio,data.shape))



if __name__ == '__main__':
    delay_time =20
    epochs = 10
    is_first_train = 1
    for i in [300,
              270,250,230,200,170,
              # 150,120,100,100,95,95,90,85,80,75,70,65,60,55,50,50,45,45,40,35,35,30,30,25,25,20
              ]:
        # 训练数据
        with open('./month_9.log','a+',encoding='utf8') as f:
            f.write('this time we will remove:{0}---------------------->>>>>>\n'.format(i))
            if is_first_train:
                df_train = pd.read_csv('./input/treb_toronto_3to8.csv')
            else:
                df_train = pd.read_csv('./input/treb_toronto_3to8_1.csv')
            # df_train = pd.read_csv('./input/treb_toronto_678_1.csv')
            # df_train = pd.read_csv('./input/treb_toronto_78_1.csv')

            # df_train = pd.read_csv('./input/treb_all_column_month_3to8.csv')
            print(df_train.shape)
            # 训练数据处理
            df_train = preprocess_data(df_train)

            # 用于预测原始数据
            df_train_middle = df_train
            df_train_prediction = df_train_middle.drop(columns=['daysOnMarket'])
            # 用于保存原始的数据
            origin_data_train = df_train.reset_index(drop=True)


            print(df_train.shape)




            # df_test_middle = pd.read_csv('./input/treb_test_month_9.csv')
            # df_test_middle = pd.read_csv('./input/treb_test_all_column_month_9.csv')
            df_test_middle = pd.read_csv('./input/treb_toronto_9.csv')
            # df_test_middle = pd.read_csv('./input/treb_toronto_10.csv')

            print(df_test_middle.shape)

            df_test_middle = preprocess_data(df_test_middle)
            print(df_test_middle.shape)
            origin_data = df_test_middle.reset_index(drop=True)


            df_train =df_train.dropna()
            df_test_middle = df_test_middle.dropna()

            df_test = df_test_middle.drop(columns='daysOnMarket')
            df_test_label = df_test_middle['daysOnMarket']

            # 数据的分离用于读入模型中
            X_train_data = df_train_prediction
            Y_train_data_label = df_train[['daysOnMarket']]
            X_test_data = df_test
            Y_test_data_label = df_test_label

            # 需要合并train和prediction的数据再拆分：不然会出现预测的时候维度不同:
            train_test_merge = pd.concat((X_train_data, X_test_data), axis=0)
            # get_dummies
            train_test_merge = pd.get_dummies(train_test_merge)
            X_train_data = train_test_merge.iloc[:X_train_data.shape[0],:]
            X_test_data = train_test_merge.iloc[X_train_data.shape[0]:,:]

            # 判断是否是第一次训练，如果是第一次训练就需要保存一次模型，如果
            # 不是就直接从本地读入模型；
            # 训练
            if is_first_train:
                model = first_train(X_train_data,Y_train_data_label,1,f)
            else:
                model = continue_train(X_train_data,Y_train_data_label,epochs,f,i)
            # 预测
            pred1 = model.predict(X_test_data)
            print(mean_absolute_error(Y_test_data_label,pred1))
            # 计算比例：
            # 合并保存测试数据预测结果
            test_data_save_and_merge_data(pred1,origin_data,f)
            # 合并保存训练数据的预测结果
            train_data_save_and_merge(model,origin_data_train,X_train_data,f)
            time.sleep(delay_time)
            data_new = pd.read_csv('./merge_data_bak/merge_train_data.csv')
            process_train_merge_data_remove_some_data(data_new,f,i)
        is_first_train=0



















