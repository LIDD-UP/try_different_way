#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: read_sql_data_to_csv_file.py
@time: 2018/7/10
"""

import pandas as pd
import pymysql

conn = pymysql.connect(host='192.168.5.201:5432', \
               user='themove.ca',password='themove.ca', \
               db='saninco_realtor_db',charset='utf8', \
               use_unicode=True)

sql = 'select * from realtor_search'
df = pd.read_sql(sql, con=conn)
print(df.head())

df.to_csv("realtor_search.csv")
conn.close()