#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: realtor_history_process.py
@time: 2018/7/17
"""

import pandas
import os
import json
import numpy as np


#文件路径：
dirname_path = os.path.dirname((os.getcwd()))
print(dirname_path)
file_path = '/dataset/realtor_history.csv'
full_path = dirname_path+file_path
print(full_path)


# 获取路径
def getpath(d, path, ans):
    if isinstance(d, str) or isinstance(d, int) or isinstance(d, bool):
        ans.append(path[0:])
    elif isinstance(d, dict):
        for i in d:
            getpath(d[i], path + '->' + str(i), ans)
    elif isinstance(d, list):
        for i in range(len(d)):
            getpath(d[i], path + '[' + str(i) + ']', ans)
    return ans


# 读取文件
realtor_data = pandas.read_csv(full_path)
realtor_data = realtor_data['realtorData']
path_list = []
count = 0
for i in realtor_data:
    count += 1
    print('总共有837413 行，已经完成%d'%count)
    if i == np.nan:
        continue
    else:
        dict_i = json.loads(i)
        # print(dict_i)
        path_list += getpath(dict_i, '', [])


print('已经完成处理，正要存入文件中。。。。。。。。。。。。')

print(len(path_list))
# 去掉重复的
path_set = set(path_list)
print(len(path_set))

# 转化成完成格式并存入到文件中
full_path_set = []
for i in path_set:
    a = 'realtor_history' + str(i)
    full_path_set.append(a)

with open('./path_list_first.text', 'w') as f:
    for i in full_path_set:
        f.writelines(i)
        f.writelines('\n')


