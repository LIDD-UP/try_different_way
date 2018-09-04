# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: columns_add_one_by_one_to_predict.py
@time: 2018/9/3
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
from scipy.special import boxcox1p,inv_boxcox1p

data = pd.read_csv('./transform_to_Gussian_remove_some_fliers.csv')



# corrmat = data.corr()
# plt.subplots(figsize=(60,30))
# sns.heatmap(corrmat, vmax=0.9, square=True,annot=True)
# plt.show()
'''
根据热度图可以看出，daysOnMarket 和其他特征的相关性几乎没有，只有少数几个有一部分的相关性；
'''




print(data.head())
# print(data.shape)
# print(data.daysOnMarket)
# data = data[data.price<1000000]
# print(data['buildingTypeId'].value_counts())
# data = data[data.buildingTypeId.isin([1,3,6])]
# data = data[data.longitude>-85]
# data = data[data.longitude<-70]
# data = data[data.bedrooms<8]
# data = data[data.longitude<50]
print(data.shape)





# 使用那几个数据最多的来进行预测：
data = data[[
    'longitude',
    'latitude',
    'price',
    'buildingTypeId',
    'bedrooms',
    'daysOnMarket',
]]

data['buildingTypeId'] = data['buildingTypeId'].astype('str')
data['bedrooms'] = data['bedrooms'].astype('str')

sns.pairplot(data)
plt.show()
def get_process_skew_numeric_feature(data):
    numeric_feats = data.dtypes[data.dtypes != "object" ].index

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


# data_sample = get_process_skew_numeric_feature(data.drop(columns='daysOnMarket'))
# data_label = pd.DataFrame(np.log1p(data['daysOnMarket']))
# data = pd.concat((data_sample,data['daysOnMarket']),axis=1)

# data = get_process_skew_numeric_feature(data)

def dummies_class_variable(data):
    data = pd.get_dummies(data)
    print(data.shape)
    return data

# data = dummies_class_variable(data)


# 预测：


if __name__ == '__main__':
    from auto_ml import Predictor
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import missingno as msno
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error

    # data =data.dropna()
    print(data.shape)
    df_train, df_test_middle = train_test_split(data,test_size=0.9)
    df_train,df_fl = train_test_split(df_train,test_size=0.1)
    df_test = df_test_middle.drop(columns='daysOnMarket')
    df_test_label = df_test_middle['daysOnMarket']

    value_list = []
    for i in range(len(data.columns)):
        value_list.append('categorical')


    column_description1 = {key:value for key in data.columns for value in value_list if data[key].dtype =='object'}
    column_description2 = {
        'daysOnMarket': 'output',
        'buildingTypeId': 'categorical'
    }

    print(column_description1)
    column_descriptions = dict(column_description1, **column_description2)


    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_train,
                       model_names='XGBRegressr'
                       )

    # ml_predictor.score(df_test)
    x = ml_predictor.predict(df_test)
    print(mean_absolute_error(inv_boxcox1p(df_test_label,0.15),inv_boxcox1p(x,0.15)))
    print(mean_absolute_error(df_test_label, x))