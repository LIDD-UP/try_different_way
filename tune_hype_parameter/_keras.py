#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: _keras.py
@time: 2018/8/6
"""
import numpy as np

np.set_printoptions(suppress=False)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# pandas 的显示设置函数：
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.set_option('max_columns', 200)
pd.set_option('display.width', 1000)

data_train_456 = pd.read_csv('./company_house_data/month_456_1.csv')
data_train_6 = pd.read_csv('./company_house_data/month_6_1.csv')
data_test_6 = pd.read_csv('./company_house_data/test_data_6_1.csv')

# 去掉buildingTypeId 为空的情况避免再编码的时候出现na这一类
data_train_456 = data_train_456[pd.isna(data_train_456.buildingTypeId) != True]
data_train_6 = data_train_6[pd.isna(data_train_6.buildingTypeId) != True]
data_test_6 = data_test_6[pd.isna(data_test_6.buildingTypeId) != True]






# 统计 count(不包括缺失值的情况）
# print(data_train_456.describe())
# print(data_train_6.describe())
# print(data_test_6.describe())
# 通过查看发现bedrooms 最大值和最小值差距有点大需要用value_counts查看一离群点；
# 接下来要对所有列进行离群点，缺失值，查看；

# data_train_456['price'].hist()
# np.log1p(data_train_456['daysOnMarket']).hist()
# plt.show()


# 接下来处理步骤：
# 1：去掉省份城市地址；
# 2：将原始数据的列调整顺序；
# 3：然后将训练数据和测试数据的标签数据和特征数据分开；
# 4：把训练数据和测试数据组合起来cancat，一起进行处理，最后再通过index取出来；
# 5: 将原本的数据中本身是类别的buildingtype转换成str，
# 6: 将原本的数值数据进行log1p处理
# 7：对buildingtypeid 进行one_hot 编码
# 8：填充缺失值
# 9：获取训练数据测试数据的最后数据；
# 10 建模处理


# 去掉省份城市地址，调整顺序
data_train_456 = data_train_456[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_train_456 shape:', data_train_456.shape)
data_train_6 = data_train_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_train_6 shape:', data_train_6.shape)
data_test_6 = data_test_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms', 'daysOnMarket']]
print('data_test_6 shape:', data_test_6.shape)

# 取出label：
data_train_456_label = data_train_456['daysOnMarket']
data_train_6_label = data_train_6['daysOnMarket']
data_test_6_label = data_test_6['daysOnMarket']


# 数据处理过程
def data_process(train, test, train_label, start_column, stop_column):
    all_data = pd.concat((train.loc[:, start_column:stop_column],
                          test.loc[:, start_column:stop_column]))

    all_data['buildingTypeId'] = all_data['buildingTypeId'].astype(str)
    print('all_data shape:', all_data.shape)

    train_label = np.log1p(train_label)

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())
    all_data = pd.get_dummies(all_data)

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train_label
    return X_train, y, X_test

# 获取处理之后的数据

# 获取train_456 的数据
train, train_label, test = data_process(data_train_6, data_test_6, data_train_6_label, 'longitude', 'bedrooms')


from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor

# 这里把训练地列数据标准化了，并没有对label数据进行标准化
# train = StandardScaler().fit_transform(train)
# # X_tr ,X_val,y_tr,y_val = train_test_split(train,train_label,random_state=3)
# # print(X_tr.shape)
# model = Sequential()
# model.add(Dense(256, activation="relu", input_dim = train.shape[1]))
# # model.add(Dense(512, activation="relu", input_dim = train.shape[1]))
# model.add(Dense(1,input_dim=train.shape[1],kernel_regularizer=l1(0.001)))
# model.compile(loss="mse",optimizer="adam")
# model.summary() # 用于总结用的，把每一层得输出形状以及参数（param）统计出来了；
# # hist = model.fit(X_tr,y_tr,validation_data=(X_val,y_val))
# model.fit(train,train_label)
# pred1 = model.predict(test)
# preds = np.expm1(model.predict(test))
# print(mean_absolute_error(data_test_6_label,preds))
# # draw = pd.DataFrame({"label":data_test_6_label,"preds":preds})
# # draw.plot(x='label',y='preds',kind='scatter')
# # pd.Series(model.predict(X_val)[:,0]).hist()
# print('preds',pred1[0:10])
# print(data_test_6_label)
# plt.plot(preds,c='blue',label='preds')
# plt.plot(data_test_6_label,c='green',label='true')
# plt.legend()
# plt.show()


X_train_data = data_train_456[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]
X_test_data = data_test_6[['longitude', 'latitude', 'price', 'buildingTypeId', 'bedrooms']]
X_train_data = StandardScaler().fit_transform(X_train_data)
X_test_data = StandardScaler().fit_transform(X_test_data)

# X_train_label = StandardScaler().fit_transform(np.array(data_train_456_label).reshape(-1,1))
# X_train_label = X_train_label.reshape(-1,1)
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


def create_model(optimizer='adam'):
    model = Sequential()
    # model.add(Dense(16, activation="relu", input_dim = X_train_data.shape[1]))
    # model.add(Dense(32, activation="relu", input_dim = X_train_data.shape[1]))
    # model.add(Dense(64, activation="relu", input_dim = X_train_data.shape[1]))
    model.add(Dense(128, activation="relu", input_dim = X_train_data.shape[1]))
    # model.add(Dense(256, activation="relu", input_dim = X_train_data.shape[1]))
    # model.add(Dense(516, activation="relu", input_dim = X_train_data.shape[1]))
    model.add(Dense(1,input_dim=X_train_data.shape[1],kernel_regularizer=l1(0.1),activation='sigmoid'))
    model.compile(loss="mse",optimizer=optimizer)
    return model
# model.summary()

model = KerasRegressor(build_fn=create_model,verbose=0)


# params = {"nb_epoch":[x for x in range(1,100)],
#           "batch_size":[x for x in range(1,1000,10)],
#             # 'optimizer':['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#           # 'shuffle':[True,False]
#           }
# model = GridSearchCV(model,param_grid=params,scoring='neg_mean_absolute_error')
# model.fit(X_train_data,data_train_456_label)
# model = model.best_estimator_


if __name__ == '__main__':
    batch_size = [ 10,20,30,40,50,60,70,80, 100,120]
    epochs = [1, 5,10,30,50,100]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # optimizer = ['SGD', 'RMSprop']
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs,optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train_data, data_train_456_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))



    model = grid_result.best_estimator_
    # model.summary()
    # 训练预测画图
    model.fit(X_train_data,data_train_456_label)
    pred1 = model.predict(X_test_data)
    # ss = StandardScaler()
    # pred1 = ss.inverse_transform(pred1)
    print(mean_absolute_error(data_test_6_label,pred1))
    print('preds',pred1[0:10])
    print(data_test_6_label)
    plt.plot(pred1[:20],c='blue',label='preds')
    plt.plot(data_test_6_label[:20],c='green',label='true')
    plt.title("kears  pre and label distribute circumstance")
    plt.legend()
    plt.show()











