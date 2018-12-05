# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: psql_tools.py
@time: 2018/11/27
"""
import psycopg2
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_conf import psql_settings
from sshtunnel import SSHTunnelForwarder
from sql_script import sql_script


class PSQLToos(object):

    def get_psql_connection_obj(self,is_ssh):
        if is_ssh:
            with SSHTunnelForwarder((psql_settings.ssh_host, psql_settings.ssh_port),
                                    ssh_password=psql_settings.ssh_password, ssh_username=psql_settings.ssh_username,
                                    remote_bind_address=(psql_settings.host, psql_settings.port)) as server:

                conn = psycopg2.connect(
                    host='localhost',
                    port=server.local_bind_port,
                    database=psql_settings.database,
                    user=psql_settings.user,
                    password=psql_settings.password
                )
                return conn
        else:
            conn = psycopg2.connect(
                host=psql_settings.host,
                port=psql_settings.port,
                database=psql_settings.database,
                user=psql_settings.user,
                password=psql_settings.password
            )
            return conn



if __name__ == '__main__':
    psql_tools = PSQLToos()
    conn = psql_tools.get_psql_connection_obj(psql_settings.is_ssh)
    print(conn)
    cursor = conn.cursor()
    query_string = sql_script.prediciton_query_string
    result = cursor.execute(query_string)
    result = cursor.fetchall()
    for i in result:
        print(result)
    conn.commit()
