# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: psql_tools.py
@time: 2018/11/27
"""
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sshtunnel import SSHTunnelForwarder
import psycopg2

from my_conf import psql_settings as jdbc
from sql_script import sql_script


class PSQLToos(object):

    def get_psql_connection_obj(self):
        conn = psycopg2.connect(
            host=jdbc.host,
            port=jdbc.port,
            user=jdbc.user,
            password=jdbc.password,
            dbname=jdbc.dbname
        )
        return conn

    def get_data(self,is_ssh):
        if is_ssh:
            with SSHTunnelForwarder((jdbc.ssh_host, jdbc.ssh_port),
                                    ssh_password=jdbc.ssh_password, ssh_username=jdbc.ssh_username,
                                    remote_bind_address=(jdbc.host, jdbc.port)) as server:

                conn = psycopg2.connect(
                    host='localhost',
                    port=server.local_bind_port,
                    database=jdbc.database,
                    user=jdbc.user,
                    password=jdbc.password
                )
                data = pd.read_sql(con=conn, sql=sql_script.prediciton_query_string)
                return data
        else:
            conn = psycopg2.connect(
                host=jdbc.host,
                port=jdbc.port,
                database=jdbc.database,
                user=jdbc.user,
                password=jdbc.password
            )
            data = pd.read_sql(con=conn,sql=sql_script.prediciton_query_string)
            return data







if __name__ == '__main__':
    psql_tools = PSQLToos()
    data = psql_tools.get_data(True)
    print(data)

