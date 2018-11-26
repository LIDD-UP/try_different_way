import pandas as pd
from sklearn import metrics

data1 = pd.read_csv('./middle.csv')
data2 = pd.read_csv('./origin_data.csv')

merge = pd.concat((data1,data2),axis=1)
print(metrics.mean_absolute_error(data1['predictions'],data2[['daysOnMarket']]))

# merge_data_df_province_data = merge[merge['province'].isin(['Ontario']) ]
#
# print('province-error-', metrics.mean_absolute_error(list(merge_data_df_province_data['daysOnMarket']),
#                                                      list(merge_data_df_province_data['predictions'])))

