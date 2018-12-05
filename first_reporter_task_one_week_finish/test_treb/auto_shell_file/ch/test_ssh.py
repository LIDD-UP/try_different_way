import psycopg2
from sshtunnel import SSHTunnelForwarder

with SSHTunnelForwarder(
         ('138.197.138.231', 22),    #B机器的配置
         ssh_password="sshpasswd",
         ssh_username="saninco123#@!",
         remote_bind_address=('138.197.138.231', 5432)) as server:  #A机器的配置

    conn = psycopg2.connect(host='127.0.0.1',              #此处必须是是127.0.0.1
                           port=server.local_bind_port,
                           user='root',
                           passwd='123456',
                           db='test3')

    conn.
    # is_ssh = True
    # ssh_host = "138.197.138.231"
    # ssh_port = 22
    # ssh_username = "root"
    # ssh_password = "saninco123#@!"
    # database = "saninco_realtor_db"
    # user = "root"
    # password = "123456"
    # host = "localhost"
    # port = 5432
    # dbname = 'test3'
