# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: raw_data_process.py
@time: 2018/8/30
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


data = pd.read_csv('./base_data_no_skew_encode.csv')
data = data[data.tradeTypeId==1]

corrmat = data.corr()
plt.subplots(figsize=(60,30))
sns.heatmap(corrmat, vmax=0.9, square=True,annot=True)
plt.show()



def get_skew_rank(data):
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness.head(40))

data = get_skew_rank(data)
# data = data.dropna()
# print(data.shape)
# print(data.head())
print(data)
print(data.dropna())
pass




'''
各种数据分布得处理方式：
    1：对于经纬度这种没有分布规律得数据：需要用到非参数检验
    2：对于price这种受极端值影响得数，去掉一部分极端值之后恢复正态分布
    3：对于rooms这种本身是数值型数据，但是却有不同得值，图形呈现一条条得竖线得情况；
    4：对于approxSquareFootage:散点图呈现条状，但是又有一点密集，hist图画出来接近连续：
        但是中间有一部分柱状图特别得高，而其余部分特别得低，
            并且通过value_counts观测出他的值情况也很少，这种目前来讲当成分类型变量比较合适
    5：这批数据存在极度不均衡的情况，这个很难下手进行处理；   
    
'''







# print(data['garageSpaces'].mode())
# print(data['lotFront'].mode())
# print(data['style'].mode())
#
# data['style'] = data['style'].fillna(str(data['style'].mode()[0]))
# print(data['style'])

'''
利用调试工具一步步的画图分析数据，去除异常值：
'''
data = data.drop(columns=[
        # 'id','propertytypeid','elevator',
        'tradeTypeId',
        # 'longitude', # 这个分布不符合正太分布，比较麻烦，长尾分布；
        # 'latitude', # 同longitude
        # 'price',
        # 'buildingTypeId',
        # 'washrooms',
        ])

def remove_numeric_flier_data(data):
    # print(len(data[data.longitude>-60]))
    data = data[data.longitude<-60]

    '''
    # 由于观察longitude的直方图发现longitude是长尾分布；有很大一部分数据都在-95到-85之间；
    对于长尾分布的数据的处理方式是截尾或者是缩尾截尾就是把尾部数据去除，
    缩尾就是将尾部数据合并成一个数据（用平均值或者是众数代替，分位数代替；平均值或者众数不太合适）
    ：也就是可以看成一种分箱操作，但是分箱其实是将数据转化成了类别型的数据；
    而前面的处理还可以将他看成连续型的数据；
    还可以使用非参数检验的方法：
    '''
    # print(len(data[data.latitude <=42])) # 50
    # print(len(data[data.latitude>=57])) # 75
    data = data[data.latitude>42]
    data = data[data.latitude<57]
    '''
    latitude 也是一个长尾分布的数据；跟longitude一样：数据的根据地理位置信息；
    应该大于43，但是由于舍弃的数据量有2000多条，暂定把界限定在42，顶点定在：57；
    这个数据呈现起伏的情况，跟longitude一样中间一部分数据很少，
    考虑要不要根据经纬度的情况，将模型分成两个部分；
    
    '''
    # print(len(data[data.price>2000000]))
    data = data[data.price<2000000]
    # data['price'] = data['price'].apply(lambda x: x if x<100000 else 1500000)
    # print(data['price'])
    '''
    price 从严格意义来件才属于长尾分布，也就是说需要将price过于离谱的数据
    去除再取1%分位点数据值作为大于1%分位点数的值，也就是利用缩尾的方法；
    但是用这种方案又导致尾部有起伏的趋势，所以还是直接用截尾的方式来做了
    但是并没有用到1%这个分位点来做，只是用了直观得观察来确定得值；
    '''
    data['buildingTypeId'] = data['buildingTypeId'].astype('str')
    data['washrooms'] = data['washrooms'].astype('str')
    '''
    由于buildingTypeId和washrooms都呈现出一种分类型变量才有的分布方式，虽然rooms本身一种
    数值型得数据，但从实际得业务出发应该处理成数值型变量，但是重模型和
    可视化得角度出发，应该处理成分类型得数据；
    对于buildingTypeId本身从实际业务来讲他就是分类型得变量；
    
    '''
    data['bedroomsPlus'] = data['bedroomsPlus'].astype('str')
    '''
    呈现分散型得状况
    '''
    # print(data['approxSquareFootage'].value_counts())
    data['approxSquareFootage'] = data['approxSquareFootage'].astype('str')

    print(data['lotFront'].value_counts())
    data = data[data.lotFront < 150]
    '''
    观察value_counts,50得有37513条；
    所以这个也无法保证正太分布得那样得规律，这样得数据
    到底是该去掉还是保留很难判断，还可以做一定得分箱；
    这个应该是我用众数填充得效果；
    但是还是要一定程度得去掉异常值
    '''


    data['kitchens']  = data['kitchens'].astype('str')

    print(data['parkingSpaces'].value_counts())
    data = data[data.parkingSpaces<30]
    '''
    我这批数据大多数都是经过缺失值填充得；
    所以很大程度上会出现不均衡得现象；但是要一定程度上得去掉异常值；
    '''

    data = data[data.room1Length<30]
    data = data[data.room1Width<50]

    data =data[data.room2Length<40]
    data = data[data.room2Width<50]

    data = data[data.room3Length < 50]
    data = data[data.room3Width < 50]

    data = data[data.room4Length < 50]
    data = data[data.room4Width < 50]

    data = data[data.room5Length < 50]
    data = data[data.room5Width < 50]

    data = data[data.room6Length < 50]
    data = data[data.room6Width < 50]

    data = data[data.room7Length < 50]
    data = data[data.room7Width < 50]

    data = data[data.room8Length < 50]
    data = data[data.room8Width < 50]

    data = data[data.room9Length < 50]
    data = data[data.room9Width < 50]

    data['rooms'] = data['rooms'].astype('str')
    '''
    根据hist图可以看出rooms图很奇特，0到2左右为一个高得柱状图，
    而右边则是呈现一种很好得正太分布情况，但是散点图呈现一种一条条得形状；
    暂时看成分类型得变量
    '''
    data  = data[data.taxes<50000]

    data['approxAge'] = data['approxAge'].astype('str')

    data = data[data.garageSpaces<5]
    data['garageSpaces'] = data['garageSpaces'].astype('str')
    '''
    暂时看成连续看一下去掉比较大得值之后会不会呈现连续得情况
    '''
    data = data[data.totalParkingSpaces<25]

    data = data[data.bedrooms<15]
    '''
    散点图呈现一条条得形状，但是hist图有一点呈现正太分布得趋势
    暂时看成连续型得变量，去掉一部分看一下能不能呈现正太分布
    '''
    data = data[data.rooms1square<1500]

    data = data[data.rooms2square<2000]

    data = data[data.rooms3square < 2500]

    data = data[data.rooms4square < 2500]

    data = data[data.rooms5square < 2500]

    data = data[data.rooms6square < 2500]

    data = data[data.rooms7square < 2500]

    data = data[data.rooms8square < 2500]

    data = data[data.rooms9square < 2000]


    return data

data = remove_numeric_flier_data(data)

# print(data['bedroomsPlus'])
# data = data.dropna()
# plt.hist(data['bedroomsPlus'])



def test_graph_of_column():
    for column in data.columns:
        if column != 'daysOnMarket' and data[column].dtype != 'object':
            plt.scatter(data[column],data['daysOnMarket'])
            plt.xlabel(column)
            plt.ylabel('daysOnMarket')
            plt.show()
            plt.hist(data[column])
            plt.show()

# test_graph_of_column()

print('finish')

print(data.describe())


data.to_csv('./transform_to_Gussian_remove_some_fliers.csv',index=False)
