#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: process_data_16000_add_column_dnn_processing.py
@time: 2018/8/21
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('./process_data_16000_add_column_dnn.csv')

print(data.head(100))
print(len(data))
# print(data['room1'][1])


# 把面积组合组合起来；
def get_square(data=data):
    rooms_colums_len =[]
    rooms_colums_wid =[]
    for i in range(1,5):
        rooms_column_str_len = 'room' + str(i) + 'Length'
        rooms_column_str_wid = 'room' + str(i) + 'Width'
        rooms_colums_len.append(rooms_column_str_len)
        rooms_colums_wid.append(rooms_column_str_wid)

    for i,len in enumerate(rooms_colums_len):
        for j,wid in enumerate(rooms_colums_wid):
            if i==j:
                name = 'rooms' + str(i + 1) + 'square'
                rooms_square_list = []
                for k in range(15266):
                    rooms_square_list_k = list(data[len])[k] * list(data[wid])[k]
                    print(rooms_square_list_k)
                    rooms_square_list.append(rooms_square_list_k)
                data[name] = rooms_square_list
                data = data.drop(columns=[len,wid])
    return data

data = get_square(data)

print(data.head(100))

data.to_csv('./generate_square.csv',index=False)




