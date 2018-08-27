# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_process.py
@time: 2018/8/27
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from random import choice

data = pd.read_csv('month6_new.csv')




# 处理bedrooms
def process_bedrooms(data):
    list_month = list(data['bedrooms'].astype('str'))
    list_month_process = []
    for i in list_month:
        if i != 'nan':
            list_month_process.append(eval(i))
        else:
            list_month_process.append(i)
    data['bedrooms'] = pd.Series(list_month_process)
    data['bedrooms'] = data['bedrooms'].astype('float')
    return data


data = process_bedrooms(data)

# 处理approxSquareFootage
def process_approxSquareFootage(data):
    data['approxSquareFootage'] = data['approxSquareFootage'].astype('str')
    list_approxS = []
    for i in data['approxSquareFootage']:
        # print(i,type(i))
        if pd.isna(i) == True:
            list_approxS.append(np.nan)
        if '-' in i :
            middle_value = i.split('-')[1]
            list_approxS.append(middle_value)
        else:
            list_approxS.append(np.nan)
    data['approxSquareFootage'] = list_approxS
    data['approxSquareFootage'] = data['approxSquareFootage'].astype('float')
    return data


data = process_approxSquareFootage(data)
# print(data['approxSquareFootage'])


# 处理approxAge
def process_approxAge(data):
    data['approxAge'] = data['approxAge'].astype('str')
    list_approxAge = []
    for i in data['approxAge']:
        # print(i,type(i))
        if pd.isna(i) == True:
            list_approxAge.append(np.nan)
        elif '-' in i :
            middle_value = i.split('-')[1]
            list_approxAge.append(middle_value)
        elif i == 'New':
            list_approxAge.append(0)
        else:
            list_approxAge.append(np.nan)
    data['approxAge'] = list_approxAge
    data['approxAge'] = data['approxAge'].astype('float')
    return data


data = process_approxAge(data)


# 将面积合并：
def get_square(data=data,num_room_number=7):
    rooms_colums_len =[]
    rooms_colums_wid =[]
    for i in range(1,num_room_number):
        rooms_column_str_len = 'room' + str(i) + 'Length'
        rooms_column_str_wid = 'room' + str(i) + 'Width'
        rooms_colums_len.append(rooms_column_str_len)
        rooms_colums_wid.append(rooms_column_str_wid)

    for i,_len in enumerate(rooms_colums_len):
        for j,wid in enumerate(rooms_colums_wid):
            if i==j:
                name = 'rooms' + str(i + 1) + 'square'
                rooms_square_list = []
                for k in range(len(data)):
                    rooms_square_list_k = list(data[_len])[k] * list(data[wid])[k]
                    print(rooms_square_list_k)
                    rooms_square_list.append(rooms_square_list_k)
                data[name] = rooms_square_list
                data = data.drop(columns=[_len,wid])
    return data
# data = get_square(data,10)
# print('square',data.shape)
# print(data.head())


data = data.drop(columns=['id','listingDate','propertytypeid','ammenitiesnearby','propertytype','community'])


def same_processing_way(data):
    data = data[data.longitude < -10]
    # data = data[data.longitude > -80]
    #
    data = data[data.latitude > 42]
    data = data[data.latitude<57]
    #
    # data = data[data.tradeTypeId == 1]
    #
    # data = data[data.price > 90000]
    # 处理price 的
    data = data[data.price < 5000000]
    # data = data[data.buildingTypeId.isin([1, 3, 6])]

    # list_bedrooms_new = []
    # for i in data['bedrooms']:
    #     if i > 6:
    #         list_bedrooms_new.append(6)
    #     else:
    #         list_bedrooms_new.append(i)
    # data['bedrooms'] = list_bedrooms_new
    #
    # data = data[data.daysOnMarket < 60]
    return data


data = same_processing_way(data)

print(data.head())
print(data.shape)

# data_feature = data.drop(columns='daysOnMarket')
# data_target = data['daysOnMarkets']

# 获取不同变量的离群点；
def remove_fliers(data):
    fig, ax = plt.subplots()
    ax.scatter(x = data['latitude'], y = data['daysOnMarket'])
    plt.ylabel('daysOnMarket', fontsize=13)
    plt.xlabel('x', fontsize=13)
    plt.show()


# 处理target variable,观察daysonMarket 的 mu和sigma值：
def get_targetVariable_mu_sigma_QQ_plt(data):
    sns.distplot(data['daysOnMarket'] , fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['daysOnMarket'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('daysOnMarket distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['daysOnMarket'], plot=plt)
    plt.show()

# get_targetVariable_mu_sigma_QQ_plt(data)

# log transform the target_variable
def log_transform_target_variable():
    data["daysOnMarket"] = np.log1p(data["daysOnMarket"])

    # Check the new distribution 
    sns.distplot(data['daysOnMarket'], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['daysOnMarket'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('daysOnMarket distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['daysOnMarket'], plot=plt)
    plt.show()

# log_transform_target_variable(data)


data['daysOnMarket'] = np.log1p(data['daysOnMarket'])
data = data[data.tradeTypeId ==1]



data.to_csv('./process_to_log.csv',index=False)
# 分开训练和测试数据
data_feature = data.drop(columns='daysOnMarket')
data_target = data['daysOnMarket']



print(data_feature.head())
print('data_feature',data_feature.shape)
print(data_target.shape)

def get_miss_ratio(data):
    data_feature_na = (data_feature.isnull().sum()/len(data_feature))*100
    data_feature_na = data_feature_na.drop(data_feature_na[data_feature_na==0].index).sort_values(ascending=False)
    missing_data= pd.DataFrame({'Missing Ratio' : data_feature_na})
    print(missing_data)
    return data_feature_na

# 绘图查看
def get_miss_plot(data_feature_na):
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='45')
    sns.barplot(x=data_feature_na.index, y=data_feature_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()

# 获取相关性矩阵
def get_corr_matrix(data):
    corrmat = data.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()


# get_miss_ratio(data)
# imputing missing values(缺失值填充）
# 具有填充意义的列：[['

# 去除掉没有意义column
data_feature = data_feature.drop(columns=['elevator','uffi','sewers',
                                          'lotSizeCode',
                                          # 'frontingOn',
                                          'lotDepth',
                                          'farmAgriculture',
                                          ])



print(data_feature['waterfront'].value_counts())
'''
None        802
Direct      144
Indirect     47
'''

data_feature['buildingTypeId'] = data_feature['buildingTypeId'].fillna(data_feature['buildingTypeId'].mode())
data_feature['furnished'] = data_feature['furnished'].fillna("None")
data_feature['approxSquareFootage'] = data_feature['approxSquareFootage'].fillna(data_feature['approxSquareFootage'].median())
data_feature['style'] = data_feature['style'].fillna(data_feature['style'].mode())
data_feature['airConditioning'] = data_feature['airConditioning'].fillna("None")
data_feature['washrooms'] = data_feature['washrooms'].fillna(0)
data_feature['bedroomsPlus'] = data_feature['bedroomsPlus'].fillna("None")
data_feature['basement1'] = data_feature['basement1'].fillna("None")
data_feature['basement2'] = data_feature['basement2'].fillna("None")
data_feature['frontingOn'] = data_feature['frontingOn'].fillna(choice(['E','W','S','N']))
data_feature['familyRoom'] = data_feature['familyRoom'].fillna(choice(['N','Y']))
data_feature['lotFront'] = data_feature['lotFront'].fillna(data_feature['lotFront'].mode())
data_feature['drive'] = data_feature['drive'].fillna("None")
data_feature['fireplaceStove'] = data_feature['fireplaceStove'].fillna("None")
data_feature['heatSource'] = data_feature['heatSource'].fillna("None")
data_feature['garageType'] = data_feature['garageType'].fillna("None")
data_feature['kitchens'] = data_feature['kitchens'].fillna(0)
data_feature['kitchensPlus'] = data_feature['kitchensPlus'].fillna("None")
data_feature['parkingSpaces'] = data_feature['parkingSpaces'].fillna(0)
data_feature['pool'] = data_feature['pool'].fillna("None")


# 对rooms进行填充缺失值
for i in range(9):
    name = 'room'+'{}'.format(str(i+1))
    # print(name)
    data_feature[name] = data_feature[name].fillna('None')  # 这个填充方法有待考虑
data_feature['rooms'] = data_feature['rooms'].fillna(0)
data_feature['taxes'] = data_feature['taxes'].fillna(data_feature['taxes'].median())
data_feature['waterIncluded'] = data_feature['waterIncluded'].fillna("N")
data_feature['approxAge'] = data_feature['approxAge'].fillna(data_feature['approxAge'].median()) # 此处用众数或者是中位数填充有待商榷；
data_feature['garageSpaces'] = data_feature['garageSpaces'].fillna(data_feature['garageSpaces'].mode())
data_feature['parkingIncluded'] = data_feature['parkingIncluded'].fillna("None")
data_feature['laundryLevel'] = data_feature['laundryLevel'].fillna("None")
data_feature['propertyFeatures3'] = data_feature['propertyFeatures3'].fillna("None")
data_feature['propertyFeatures4'] = data_feature['propertyFeatures4'].fillna("None")
data_feature['propertyFeatures5'] = data_feature['propertyFeatures5'].fillna("None")
data_feature['propertyFeatures6'] = data_feature['propertyFeatures6'].fillna("None")
data_feature['waterfront'] = data_feature['waterfront'].fillna("None")
data_feature['totalParkingSpaces'] = data_feature['totalParkingSpaces'].fillna(0)
data_feature['bedrooms'] = data_feature['bedrooms'].fillna(data_feature['bedrooms'].mode())
data_feature['ownershiptype'] = data_feature['ownershiptype'].fillna(data_feature['ownershiptype'].mode())


print(data_feature.dtypes)
print(data_feature['ownershiptype'].value_counts())




