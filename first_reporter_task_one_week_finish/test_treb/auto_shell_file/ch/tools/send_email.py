# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: send_email.py
@time: 2018/11/28
"""
import smtplib
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from email.mime.text import MIMEText
import my_conf.emial_settings as mail_conf

_TITLE = 'Production 机器预测状态: '


def send_email(title='', content=''):
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = "{}".format(mail_conf.sender)
    message['To'] = ",".join(mail_conf.receivers)
    message['Subject'] = _TITLE+title
    try:
        smtp_obj = smtplib.SMTP_SSL(mail_conf.mail_host, 465)
        smtp_obj.login(mail_conf.mail_user, mail_conf.mail_pass)
        smtp_obj.sendmail(mail_conf.sender, mail_conf.receivers, message.as_string())
        print("mail has been send successfully.")
    except smtplib.SMTPException as e:
        print(e)


if __name__ == '__main__':
    send_email('AI 预测','Sucessful')