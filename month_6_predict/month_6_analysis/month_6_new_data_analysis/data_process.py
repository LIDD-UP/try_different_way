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


'''
process 的处理步骤：
 1：将bedrooms eval
 2：# 处理approxSquareFootage，# 处理approxAge，取最大值由于这两个数是一个范围，我这里只取了这两个数的最大值；但是看来，两个数的均值更加的可取；
 3：特征衍生，合并room的面积
 4：drop掉一些无用的特征
 5：处理一些连续特征得离群点
 6：再drop掉一些无用特征
 7：填充缺失值
 8：将离散得数据标签编码
 9：将连续得skew（）得特征进行box-cox转化
 10：get_dummies
 
'''

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

# 处理bedrooms(eval)
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

# 处理square；取最大值；
data = process_approxSquareFootage(data)
# print(data['approxSquareFootage'])


# 处理approxAge，取最大值
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
                # data = data.drop(columns=[_len,wid]) # 这里先不drop掉，特征组合和衍生是对特征扩充，不应该直接删除，
                # 先保留，调试之后看结果；
    return data
data = get_square(data,10)
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
    data = data.reindex()
    return data


data = same_processing_way(data)

print(data.head(50))
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
    daysOnMarket_skew = data['daysOnMarket'].skew()
    sns.distplot(data['daysOnMarket'] , fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['daysOnMarket'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} ,{})'.format(mu, sigma,daysOnMarket_skew)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('daysOnMarket distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['daysOnMarket'], plot=plt)
    plt.show()

# get_targetVariable_mu_sigma_QQ_plt(data)

# log transform the target_variable
def log_transform_target_variable(data):
    # log 和 sqrt 这两种方法还是要看实际效果来决定；
    daysOnMarket_skew_before = data['daysOnMarket'].skew()
    # data["daysOnMarket"] = np.log1p(data["daysOnMarket"])
    data['daysOnMarket'] = np.sqrt(data['daysOnMarket'])
    daysOnMarket_skew_after = data['daysOnMarket'].skew()
    # Check the new distribution 
    sns.distplot(data['daysOnMarket'], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['daysOnMarket'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} {}{})'.format(mu, sigma,daysOnMarket_skew_before,daysOnMarket_skew_after)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('daysOnMarket distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['daysOnMarket'], plot=plt)
    plt.show()

# log_transform_target_variable(data)


# data['daysOnMarket'] = np.log1p(data['daysOnMarket'])
# data['daysOnMarket'] = np.sqrt(data['daysOnMarket'])
data = data[data.tradeTypeId ==1]
# sns.distplot(data['daysOnMarket'])
# plt.show()



# data.to_csv('./process_to_log.csv',index=False)
# 分开训练和测试数据
# data_feature = data.drop(columns='daysOnMarket')
data_feature = data
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

data_feature['buildingTypeId'] = data_feature['buildingTypeId'].fillna(data_feature['buildingTypeId'].mode()[0])
data_feature['furnished'] = data_feature['furnished'].fillna("None")
data_feature['approxSquareFootage'] = data_feature['approxSquareFootage'].fillna(data_feature['approxSquareFootage'].median())
data_feature['style'] = data_feature['style'].fillna(data_feature['style'].mode()[0])
print(data_feature['style'])
data_feature['airConditioning'] = data_feature['airConditioning'].fillna("None")
data_feature['washrooms'] = data_feature['washrooms'].fillna(0)
data_feature['bedroomsPlus'] = data_feature['bedroomsPlus'].fillna("None")
data_feature['basement1'] = data_feature['basement1'].fillna("None")
data_feature['basement2'] = data_feature['basement2'].fillna("None")
data_feature['frontingOn'] = data_feature['frontingOn'].fillna(choice(['E','W','S','N']))
data_feature['familyRoom'] = data_feature['familyRoom'].fillna(choice(['N','Y']))
data_feature['lotFront'] = data_feature['lotFront'].fillna(data_feature['lotFront'].mode()[0])
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
data_feature['garageSpaces'] = data_feature['garageSpaces'].fillna(0) # 车库应该用0填充，由于它本身是具有意义的；
data_feature['parkingIncluded'] = data_feature['parkingIncluded'].fillna("None")
data_feature['laundryLevel'] = data_feature['laundryLevel'].fillna("None")
data_feature['propertyFeatures3'] = data_feature['propertyFeatures3'].fillna("None")
data_feature['propertyFeatures4'] = data_feature['propertyFeatures4'].fillna("None")
data_feature['propertyFeatures5'] = data_feature['propertyFeatures5'].fillna("None")
data_feature['propertyFeatures6'] = data_feature['propertyFeatures6'].fillna("None")
data_feature['waterfront'] = data_feature['waterfront'].fillna("None")
data_feature['totalParkingSpaces'] = data_feature['totalParkingSpaces'].fillna(0)
data_feature['bedrooms'] = data_feature['bedrooms'].fillna(data_feature['bedrooms'].mode()[0]) # bedrooms应该是有的，本身情况也是缺失不多；
data_feature['ownershiptype'] = data_feature['ownershiptype'].fillna(data_feature['ownershiptype'].mode()[0]) # 这是拥有者的类型，是拥有权还是出租权
print(data_feature.dtypes)
# 对room进行缺失值填充，并将roomi为none的面积设置为0

print(data_feature.head(50))
data_feature = data_feature.reindex([x for x in range(len(data_feature))])
print(data_feature.head(50))
print(data_feature['room1'].loc[9])

def fill_roomi_na(data,room_num):
    for i in range(len(data)):
        for j in range(room_num):
            room_name = 'room' + '{}'.format(str(j + 1))
            room_len_name = 'room' + '{}'.format(str(j + 1)) + 'Length'
            room_wid_name = 'room' + '{}'.format(str(j + 1)) + 'Width'
            room_square_name = 'rooms' + '{}'.format(str(j + 1)) + 'square'
            if data[room_name].loc[i] == 'None':
                print(i,j)
                # print(data[room_len_name].loc[i])
                # print(data[room_wid_name].loc[i])
                data[room_len_name].loc[i] = 0
                data[room_wid_name].loc[i] = 0
                # print(data[room_len_name].loc[i])
                # print(data[room_wid_name].loc[i])
                data[room_square_name].loc[i] = 0
    return data


data_feature = fill_roomi_na(data_feature,9)
# print(data_feature.shape)


# more feature engineering : transform the numeric class feature to  str ;
data_feature['buildingTypeId'] = data_feature['buildingTypeId'].astype('str')

def label_encode(data):
    for column in data.columns:
        if data[column].dtypes=='object':
            data[column] = pd.factorize(data[column].values, sort=True)[0] + 1
            data[column] = data[column].astype('str')
    return data


# data_feature = label_encode(data_feature)

# 合并数据：
# data_merge_feature_target = pd.concat((data_feature,data_target))
# data_feature.to_csv('./processing_missing.csv', index=False)


# 倾斜的skew>0.75的连续型特征进行box-cox变换；

# 查看倾斜的特征
def get_process_skew_numeric_feature(data):
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)

    # 将处理skew的特征
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # data[feat] += 1
        data[feat] = boxcox1p(data[feat], lam)

    # data[skewed_features] = np.log1p(data[skewed_features])
    return data


# data_feature = get_process_skew_numeric_feature(data_feature)


# get_dummies class variable
def dummies_class_variable(data):
    data = pd.get_dummies(data)
    print(data.shape)
    return data


# data_feature = dummies_class_variable(data_feature)

data_feature.to_csv('./base_data_no_skew_encode.csv', index=False)


# label_encode 和getdummies()是非必要的步骤，先将数据运行完了之后再结合模型考虑哟啊不要进行get_dummies或者是labelencode；