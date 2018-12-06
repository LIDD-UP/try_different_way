# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: data_process.py
@time: 2018/11/27
"""
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from my_conf.merge_data_file_for_dummies_settings import merge_data_path
# print('merege_data_path:',merge_data_path)


class DataProcess(object):

    def data_process(self,data,predict_or_test):
        '''
        this function is be used to process data which read from database
        :param
        data:must be pandas.DataFrame
        '''
        # predict_or_test 的区别在于，test在处理的时候需要用daysOnMarket,zhe个用于保存
        # 原始数据，并且计算误差和预测比例
        if predict_or_test == 'test':
            data = data[[
                'estateMasterId',
                'realtorDataId',
                'realtorHistoryId',
                'mlsNumber',
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
            data['price'] = data['price'].astype(int)
            data['buildingTypeId'] = data['buildingTypeId'].astype(int)
            data['tradeTypeId'] = data['tradeTypeId'].astype(int)
        else:
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
                # 'daysOnMarket',
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
            data['price'] = data['price'].astype(int)
            data['buildingTypeId'] = data['buildingTypeId'].astype(int)
            data['tradeTypeId'] = data['tradeTypeId'].astype(int)

        return data


    def keras_merge_data_process(self,data):
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
            # 'daysOnMarket',
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
        data['price'] = data['price'].astype(int)
        data['buildingTypeId'] = data['buildingTypeId'].astype(int)
        data['tradeTypeId'] = data['tradeTypeId'].astype(int)
        return data



    def keras_data_process(self,data):
        merge_data = pd.read_csv(merge_data_path)
        merge_data = self.keras_merge_data_process(merge_data)

        # 这里有一个合并问题：需要去除daysOnMarket
        # merge_data = merge_data.drop(columns='daysOnMarket')

        print('merge_dummies_data:',merge_data.head())
        print('merge_dummies_data:',merge_data.shape)
        # 这里再合并得时候要注意字段名称得问题；
        new_data = pd.concat((data,merge_data))
        print('concat shape:',new_data.shape)
        # 这里dummies得时候，原始得tradeTypeId和buildingTypeId 没有转化成类别型得变量；
        new_data = pd.get_dummies(new_data)
        final_data = new_data.iloc[:data.shape[0],:]
        print('get_dummies shape:',final_data.shape)
        return final_data


if __name__ == '__main__':
    db = DataProcess()
    # 这里需要注意合并时候字段名称得问题i；
    test_dataframe = pd.DataFrame([[1,1,1,1,1,1,1,1,1,1]])
    print(test_dataframe.shape)
    db.keras_data_process(test_dataframe)

