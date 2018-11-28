# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_merge_dummies_data.py
@time: 2018/11/28
"""
import pandas as pd


class GetMergeDummiesData(object):
    def get_merge_dummies_data(self,data,category_columns):
        '''
        data:merge data for get_dummies ;must be DataFrame type;
        category_columns:category columns ,must be list;
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
        ]]
        data = data.dropna()
        new_data = pd.DataFrame()
        print(data.shape)
        for column in data.columns:
            # 常规套路
            # if column in category_columns:
            # 特殊方式
            # 这里还可以优化：
            if column in [
                "tradeTypeId",
                "buildingTypeId",
                'ownerShipType',
                'district']:
                new_data = pd.concat((pd.DataFrame(list(set(list(data[column]))),columns=[column]), new_data), axis=1)
        #

        print(new_data.shape)
        print(new_data)
        new_data.to_csv('../merge_data_for_dummies/merge_data_for_dummies.csv')


if __name__ == '__main__':
    data = pd.read_csv('../merge_data_for_dummies/solve_dummies_problem.csv')
    get_merge_dummies_data_obj = GetMergeDummiesData()
    get_merge_dummies_data_obj.get_merge_dummies_data(data,[])
