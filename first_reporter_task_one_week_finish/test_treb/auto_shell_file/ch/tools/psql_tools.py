# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: psql_tools.py
@time: 2018/11/27
"""
import psycopg2
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.curdir)))
from Configure import psql_settings



class PSQLToos(object):
    def __init__(self):
        self.host = psql_settings.HOST
        self.port = psql_settings.PORT
        self.user = psql_settings.USER
        self.password = psql_settings.PASSWORD
        self.dbname = psql_settings.DBNAME
    def get_psql_connection_obj(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname
        )
        return conn


if __name__ == '__main__':
    psql_tools = PSQLToos()
    conn = psql_tools.get_psql_connection_obj()
    print(conn)
