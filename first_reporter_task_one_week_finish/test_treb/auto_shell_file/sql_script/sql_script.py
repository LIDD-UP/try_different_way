# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: sql_script.py
@time: 2018/11/27
"""

import psycopg2
import pandas as pd
from sklearn.metrics import mean_absolute_error
from auto_ml import load_ml_model


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
        # 'style', #22.769283885157083
        # 'community', # 类似于city类型得数据，类型有766个； #22.38147912725983
        # 'airConditioning', #22.755048806968883
        # 'washrooms', # 连续 #23.691205780782205
        # 'basement1',# 地下室22.797430800725444
        # 'familyRoom', # 22.794731300998404
        # 'fireplaceStove', # 2 w 左右 #22.82878318024665
        # 'heatSource', # 数据量可以2w+# 22.75554140962404
        # 'garageType', # 2 w+ #22.79707321027956
        # 'kitchens', # 22.79393809434976
        # 'parkingSpaces', #22.807931672409705
        # #
        # 'parkingIncluded',#22.786586056260784
        # 'rooms',# 22.785397232054713
        #
        # 'waterIncluded', # 22.80653144493355
        # 'totalParkingSpaces', # 22.81551411353129
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
    return data



if __name__ == '__main__':
    # 导入数据
    # data = pd.read_csv('./input/treb_toronto_9.csv')
    # data = preprocess_data(data)
    # origin_data = data.reset_index(drop=True)
    #
    # data_label = data['daysOnMarket']
    # data_prediction = data.drop(columns=['daysOnMarket'])
    # 导入模型
    # model = load_ml_model('./some_fine_model/model_auto_ml_9.h5')
    # 预测：
    # prediction_result = model.predict(data_prediction)

    # x = prediction_result
    # x_dataframe = pd.DataFrame(x, columns=['predictions'])
    # merge_data = pd.concat((origin_data, x_dataframe), axis=1)
    # merge_data_df = pd.DataFrame(merge_data)
    # merge_data_df.to_csv('./intermediate_generated_file/load_model_test_merge_prediction_result_data.csv', index=False)
    # print(x_dataframe.describe())
    # print(data_label.describe())
    #
    # print(mean_absolute_error(data_label, x))
    # compute_ratio(merge_data_df)
    conn = psycopg2.connect(
        host='127.0.0.1',
        port='5432',
        database='test',
        user='postgres',
        password='123456'
    )
    cur = conn.cursor()
    sql_query_string = '''
        SELECT
        rs."id",
        rs.province,
        rs.city,
        rs.address,
        rs."postalCode",
        TRIM (rs.longitude) :: NUMERIC AS longitude,
        TRIM (rs.latitude) :: NUMERIC AS latitude,
        rs.price,
        rs."buildingTypeId",
        rs."tradeTypeId",
        rs."listingDate",
        
        rs.furnished,
        rs."style",
        rs.community,
        rs."airConditioning",
        rs.washrooms,
        rs.basement1,
        rs."familyRoom",
        rs."fireplaceStove",
        rs."heatSource",
        rs."garageType",
        rs.kitchens,
        rs."parkingSpaces",
        rs."parkingIncluded",
        rs.rooms,
        rs."waterIncluded",	
        rs."totalParkingSpaces",
        rs.district,
        rs."daysOnMarket",
        rs."bedrooms",
        rs."bathroomTotal"
    FROM
        realtor_data rs
    WHERE
        1 = 1
    AND rs.latitude IS NOT NULL
    AND rs.longitude IS NOT NULL
    AND rs.latitude != ''
    AND rs.longitude != ''
    AND rs.city = 'Toronto'

    '''
    # cur.execute(sql)
    # sql_dataframe = pd.DataFrame(cur.fetchall())
    # print(sql_dataframe.head())
    data = pd.read_sql(sql_query_string,conn)
    print(data.head())
    # for result in cur.fetchall():
    #     print(result)

    # conn.commit()