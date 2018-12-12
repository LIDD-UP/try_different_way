#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: different_processing_way.py
@time: 2018/7/13
"""

'''
对于特定特征，有数值的个数大于10000保留；需要用到pd_isnull_funcion里面的获取数据不为空的长度
'''

import pandas as pd
import os
import re



def read_csv_data(filename):
    # 这种方式的文件读取方式仅限于py文件和dataset 文件属于同一级目录下
    current_path = os.getcwd() #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    data = pd.read_csv(file_path)
    return data


def to_csv_file(dataframe,filename):
    # 把dataframe转换成csv文件并存储到dataset下,仅限于py文件和dataset文件属于同一级目录下
    current_path = os.getcwd()  #
    file_name = '/dataset/{}.csv'.format(filename)
    file_path = current_path + file_name
    dataframe.to_csv(file_path, index=False)




'''
# 取出列有效数据大于10000 的列
# data = read_csv_data('realtor_data')
# print(data)
# def get_column_based_on_nan_len(dataframe,length):
#     new_data = pd.DataFrame()
#     for column in dataframe.columns:
#         if len(dataframe[column]) != 0:
#             column_is_null = pd.isnull(dataframe[column])
#             column_is_null_true = column_is_null[column_is_null]
#             column_is_null_len = len(column_is_null_true)
#             column_not_null_len = len(dataframe[column])-column_is_null_len
#             if column_not_null_len > length:
#                 new_data[column] = dataframe[column]
#     return new_data
# 
# #获取不为空数据量大于10000的数据
# new_data = get_column_based_on_nan_len(data,10000)
# print('new_data',len(new_data.columns))
# to_csv_file(new_data,'more_than_10000')
'''
# ---------------------------------------------->>>>

'''
# 取出列有效数据大于10000 但是缺失比率大于%50，也就是说这时候需要找到缺失率大于%50 的列；
def get_column_based_on_ration(dataframe,ratio):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        if len(dataframe[column]) != 0:
            column_is_null = pd.isnull(dataframe[column])
            column_is_null_true = column_is_null[column_is_null]
            column_is_null_len = len(column_is_null_true)
            if column_is_null_len / len(dataframe[column]) > ratio:
                new_data[column] = dataframe[column]
    return new_data

new_data = read_csv_data('more_than_10000')
new_data_processing = get_column_based_on_ration(new_data,0.5)
print(len(new_data_processing.columns))
to_csv_file(new_data_processing,'more_10000_and_more_middle')
'''


# --------------------------------------->>>>

'''
# 取出缺失值小于%50的
def get_column_based_on_ration(dataframe,ratio):
    new_data = pd.DataFrame()
    for column in dataframe.columns:
        if len(dataframe[column]) != 0:
            column_is_null = pd.isnull(dataframe[column])
            column_is_null_true = column_is_null[column_is_null]
            column_is_null_len = len(column_is_null_true)
            if column_is_null_len / len(dataframe[column]) <= ratio:
                new_data[column] = dataframe[column]
    return new_data

new_data = read_csv_data('more_than_10000')
new_data_processing = get_column_based_on_ration(new_data,0.5)
print(len(new_data_processing.columns))
to_csv_file(new_data_processing,'less_than_middle')
'''

# ------------------------------------->>>>

'''
# 以缺失值在%50以上，有效数据大于10000中的列为基准在有效数据大于10000上获取有效数据：

# 获取列：

more_than_10000_data = read_csv_data('test_house_info')
# more_10000_and_more_middle_data = read_csv_data('more_10000_and_more_middle')
# based_column = more_10000_and_more_middle_data.columns
# print(based_column[0])
# contactFirstName

# 还是要拿more_than_10000 数据，以10000有效数据到，缺失率大于%50 的列（不为空的）为基准获取样本数据；


def delete_row_based_on_column_null(dataframe,column):
    new_data = pd.DataFrame()
    count = 0  # 第几次删除行，第一次就要用原始数据进行赋值，后面就不用了，直接inplace=True，为了保留原始文件；
    for i in range(len(dataframe)):
            row_i = dataframe.loc[i]
            # 判断指定位置的数书否缺失
            column_row_i = row_i[column]
            if pd.isna(column_row_i):
                count += 1
                if count == 1:
                    new_data = dataframe.drop(index=i)
                    print('new_data_len_i', len(new_data))
                if count > 1:
                    new_data.drop(index=i, inplace=True)
    return new_data


new_data = delete_row_based_on_column_null(more_than_10000_data,'price')
print(more_than_10000_data.shape)
print(new_data.shape)


# # 循环处理 获取每一次处理之后的文件；
# for column in more_10000_and_more_middle_data.columns:
#     new_data = delete_row_based_on_ration(more_than_10000_data,column)
#     print(column,':的形状',new_data.shape)
#     new_data.tocsv(new_data,'{}'.format(column))
'''

# ------------------------>>>>


'''
# 去除掉特征名包含某个单词或者是以什么开头，以什么结尾的列；
# 这里就去掉以Flag，mlsNumber  delislingDate postalCode processesAddress updateTimestamp结尾的；

data = read_csv_data('more_than_10000')
new_data = pd.DataFrame()
for column in data.columns:
    find_Flag = re.findall('Flag$',column)
    find_Number = re.findall('Number$',column)
    find_stamp = re.findall('stamp$',column)
    find_processes = re.findall('^processes',column)
    find_expected = re.findall('^expected',column)

    if len(find_Flag) == 0 and len(find_Number) == 0 and len(find_stamp) == 0 and len(find_processes) == 0 and len(find_expected) ==0:
        print(' not find ')
        new_data[column] = data[column]

print(len(data.columns))
print(len(new_data.columns))
to_csv_file(new_data,'re_delete')

'''
# -------------------------------------------->>>>>>>

feature_import = '''
address,province, city, price, 
buildingTypeId,tradeTypeId ,listingDate ,furnished,
approxSquareFootage, trebStatus ,style ,municpCode,
communityCode, municipalityDistrict ,municipality ,
community,airConditioning ,acreage,washrooms ,bedrooms ,
bedroomsPlus,basement1,basement2 ,cableTVIncluded,cacIncluded ,
commonElementsIncluded  ,frontingOn  ,directions,familyRoom ,lotDepth,
drive ,utilitiesHydro,extras, fireplaceStove ,lotFront, 
heatSource,garageType ,utilitiesGas ,heatType, lotIrregularities ,
kitchens, parkingSpaces,pool,parkingIncluded ,listBrokerage,
room1Length,room1,room1Width,room2Length,room2,room2Width,
room3Length,room3,room3Width,room4Length,room4,room4Width,
room5Length,room5,room5Width,room6Length,room6,room6Width,
room7Length,room7,room7Width,room8Length,room8,room8Width,
room9Length,room9,room9Width,rooms,sewers,streetName,
streetDirection,streetNo,taxes,uffi,waterIncluded,
washroomsType1Pcs,washroomsType2Pcs,washroomsType3Pcs,
washroomsType4Pcs,washroomsType1,washroomsType2,washroomsType3,
washroomsType4,taxYear,approxAge,zoning,typeOwnSrch,typeOwn1Out,
exterior1,exterior2,otherStructures1,garageSpaces,laundryAccess,
privateEntrance,kitchensPlus,laundryLevel,propertyFeatures3,propertyFeatures4,
propertyFeatures5,propertyFeatures6,retirement,waterfront,specialDesignation1,
parcelOfTiedLand,totalParkingSpaces,district,latitude,longitude
'''
a_list_true =[]
a_list_1 = feature_import.split(',')
for i in a_list_1:
    i = i.strip()
    i = i.replace('\n','')
    # print(i)
    if i == 'price':
        print('find')
    a_list_true.append(i)
# print(a_list_true)
import_feature_list = a_list_true

data = read_csv_data('human_judge')
print(len(data))


def delete_row_based_on_column_null(dataframe,column):
    new_data = pd.DataFrame(columns=import_feature_list)
    data_length = len(dataframe)
    for i in range(data_length):
            row_i = dataframe.loc[i]
            # 判断指定位置的数书否缺失
            column_row_i = row_i[column]
            if pd.notna(column_row_i):
                new_data.loc[i] = dataframe.loc[i]
                print('new_data_len_i', len(new_data))
    return new_data


new_data = delete_row_based_on_column_null(data, 'furnished')
print(new_data.shape)
print(data.shape)
to_csv_file(new_data, 'based_on_furnished')


