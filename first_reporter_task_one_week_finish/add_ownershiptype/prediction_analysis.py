import pandas as pd
pd.set_option('display.column',100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

data = pd.read_csv('./merge_data_auto_ml.csv')

data_10 = data[(abs(data.daysOnMarket-data.predictions))<10]
print(len(data_10)/len(data))
# print(data_10[['daysOnMarket','predictions']])

data_20  = data[(abs(data.daysOnMarket-data.predictions))<20]
print(len(data_20)/len(data))




data_30  = data[(abs(data.daysOnMarket-data.predictions))<=30]
print(len(data_30)/len(data))

data_more_30  = data[(abs(data.daysOnMarket-data.predictions))>30]
print(len(data_more_30)/len(data))


# data_10 = []
# data_20 = []
# data_30 = []
# data_more = []
#
# for i in range(len(data)):
#     print(i)
#     if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) <=10:
#         data_10.append(i)
#     if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) >10 and abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket'])<=20:
#         data_20.append(i)
#     if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) >20 and abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket'])<=30:
#         data_30.append(i)
#     if abs(data.iloc[i]['predictions'] - data.iloc[i]['daysOnMarket']) >30:
#         data_more.append(i)
#
# print(len(data_10)/len(data))
# print(len(data_20)/len(data))
# print(len(data_30)/len(data))
# print(len(data_more)/len(data))