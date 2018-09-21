# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_process.py
@time: 2018/9/21
"""
import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
color = sns.color_palette()

train_df = pd.read_csv('../input/month_567_data_process_1.csv')
test_df = pd.read_csv('../input/hose_info_201808_predict_2.csv')
train_df = train_df.drop(columns=['记录数','id'])
print(train_df.head())
print(train_df.shape)
train_df = train_df.dropna()
print(train_df.shape)
# train_df = train_df.drop(columns=['address','postalCde','delistingDate','listingDate'])

# 首先观察target feature
def observe_target_feature(train_df):
    int_level = train_df['daysOnMarket'].value_counts()

    plt.figure(figsize=(8,4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('daysOnMarket', fontsize=12)
    plt.show()

# observe_target_feature(train_df)
'''
可以看出daysOnMarket是有一点像正态分布的，两边低，中间高的形态，但是
看起来更像是向左偏的，有一点像长尾分布；
'''

# 观察bathroomsTotal
def observe_bathroomstotal(train_df):
    # 将bathroomTotal>5的都看作是5
    train_df['bathroomTotal'].ix[train_df['bathroomTotal'] > 5] = 5

    cnt_srs = train_df['bathroomTotal'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('bathroomTotal', fontsize=12)
    plt.show()
'''
bathroomTotal 也是呈现类似正太分布的，
但是更偏左，而且右边更像是长尾分布；
'''


# observe_bathroomstotal(train_df)

# # 将bathroomTotal>5的都看作是5，观察他在daysOnMarket上的分布状况；（小提琴图）
# 这个小提琴图不太适合用在这；适合target是分类型的情况；
def process_bathroomtatal(train_df):
    train_df['bathroomTotal'].ix[train_df['bathroomTotal'] > 5] = 5
    plt.figure(figsize=(8, 4))
    sns.violinplot(x='daysOnMarket', y='bathroomTotal', data=train_df)
    plt.xlabel('daysOnMarket', fontsize=12)
    plt.ylabel('bathroomTotal', fontsize=12)
    plt.show()

# process_bathroomtatal(train_df)

# barplot 条形图
def observe_bedrooms_by_barplot(train_df):
    cnt_srs = train_df['bedrooms'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('bedrooms', fontsize=12)
    plt.tight_layout()
    plt.show()
'''
bedrooms 就是一个长尾分布的形态，随着bedrooms的变多，样本量越少；
'''
# observe_bedrooms_by_barplot(train_df)

# 这个countplot用在这不太合适
def observe_bedrooms_by_countplot(train_df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='bedrooms', hue='daysOnMarket', data=train_df)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('bedrooms', fontsize=12)
    plt.tight_layout()
    plt.show()

# observe_bedrooms_by_countplot(train_df)

# 观察price的分布；可以看出他有outliers；把outliers去除之后再看；
def observe_price_distribute(train_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df.price.values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('price', fontsize=12)
    plt.show()


# observe_price_distribute(train_df)
# 但是用99% 的数代替大于他的数，分布不好，看来只能去除太大的数；
def process_price_and_observe_price_distribute(train_df):
    # 取出%99的数
    ulimit = np.percentile(train_df.price.values, 99)
    # 把大于99%的数据设置成99%的分位数
    train_df['price'].ix[train_df['price'] > ulimit] = ulimit
    # train_df['price'] = train_df['price'].dropna()

    plt.figure(figsize=(8, 6))
    # sns.distplot(train_df.price.values, bins=10000, kde=True)
    # plt.hist(train_df['price'])
    plt.xlabel('price', fontsize=12)
    plt.tight_layout()
    plt.show()


# process_price_and_observe_price_distribute(train_df)

# 尝试用盒图去除outliers

def remove_filers_with_boxplot(data):
    # data_columns = [column for column in data.columns if data[column].dtype!='object']
    data_columns = ['price']
    p = data.boxplot(return_type='dict')
    for index,value in enumerate(data_columns):
        print(value)
        # 获取异常值
        fliers_value_list = p['fliers'][index].get_ydata()
        print(fliers_value_list)
        print(len(fliers_value_list))
        # 删除异常值
        for flier in fliers_value_list:
            print(flier)
            data = data[data.loc[:,value] != flier]
    return data

# print(train_df.shape)
# train = remove_filers_with_boxplot(train_df)
# print(train.shape)

def get_column_outliers(data):
    p = data.boxplot(return_type='dict')
    fliers_value_list = p['fliers'][6].get_ydata()
    print(fliers_value_list)
    print(len(fliers_value_list))
    # 删除异常值
    for flier in fliers_value_list:
        print(flier)
        data = data[data.loc[:, 'price'] != flier]
    return data
# print(train_df.shape)
# train = get_column_outliers(train_df)

# print(train.shape)
# train.to_csv('./process_fliers.csv',index=False)


def plot_data(train_df):
    sns.pairplot(train_df)
    plt.tight_layout()
    plt.show()


# plot_data(train_df)

print(len(train_df[train_df.bathroomTotal>1000]))

# 分析数据进行处理
def observe_pairplot_to_process_data(train_df):
    train_df = train_df[train_df.bathroomTotal<1000]
