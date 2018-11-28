# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_process.py
@time: 2018/11/27
"""
import pandas as pd


class DataProcess(object):

    @classmethod
    def data_process(self,data):
        '''
        this function is be used to process data which read from database
        :param
        data:must be pandas.DataFrame
        '''
        data = data[[
            "longitude",
            "latitude",
            # "city",
            # "province",
            "price",
            "tradeTypeId",
            # # "listingDate",
            "buildingTypeId",
            "bedrooms",
            "bathroomTotal",
            # 'postalCode',
            'daysOnMarket',
            'ownerShipType',
            # 'projectDaysOnMarket',
            'district',

            # 以下就是用于测试得新得特征；
            # 'style', #22.769283885157083
            # 'community', # 类似于city类型得数据，类型有766个； #22.38147912725983
            # 'airConditioning', #22.755048806968883
            # 'washrooms', # 连续 #23.691205780782205
            # 'basement1',# 地下室22.797430800725444
            # 'familyRoom', # 22.794731300998404
            # 'fireplaceStove', # 2 w 左右 #22.82878318024665
            # 'heatSource', # 数据量可以2w+# 22.75554140962404
            # 'garageType', # 2 w+ #22.79707321027956
            # 'kitchens', # 22.79393809434976
            # 'parkingSpaces', #22.807931672409705
            # #
            # 'parkingIncluded',#22.786586056260784
            # 'rooms',# 22.785397232054713
            #
            # 'waterIncluded', # 22.80653144493355
            # 'totalParkingSpaces', # 22.81551411353129
            #
            # 'frontingOn',  # 面向得方向，drop掉之后有1w多:14270
            # 'drive',  # 14270
            # 'pool',  # 这个偏少；14270
            # 'sewers',  # 这个数据比较少 1w+：14270

            # more column
            # 'room3',

        ]]
        # 是否只考虑tradeTypeId为1的数据
        # data = data[data.tradeTypeId == 1]
        # data = data.drop(columns=['tradeTypeId'])
        print(data.shape)
        data = data.dropna(axis=0)
        print(data.shape)
        bedrooms_list = []
        for bedrooms in data["bedrooms"]:
            # print(bedrooms)
            if isinstance(bedrooms, float):
                bedrooms_list.append(int(bedrooms))
            elif isinstance(bedrooms, int):
                bedrooms_list.append(int(bedrooms))
            else:
                bedrooms_list.append(int(eval(bedrooms)))
        data["bedrooms"] = bedrooms_list
        bathroom_total_list = []
        for bathroom_total in data["bathroomTotal"]:
            bathroom_total_list.append(int(bathroom_total))
        data["bathroomTotal"] = bathroom_total_list

        # 将price做log变换
        # data['price'] = np.log1p(data['price'])
        return data

    def keras_data_process(self,data):
        # 这里用相对路径还是有问题；
        merge_data = pd.read_csv('../merge_data_for_dummies/merge_data_for_dummies.csv')
        new_data = pd.concat((data,merge_data))
        new_data = pd.get_dummies(new_data)
        final_data = new_data.iloc[:data.shape[0],:]
        print(final_data.shape)
        return final_data


if __name__ == '__main__':
    db = DataProcess()
    db.keras_data_process([])

