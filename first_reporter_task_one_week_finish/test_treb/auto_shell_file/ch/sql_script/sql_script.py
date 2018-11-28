# -*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: sql_script.py
@time: 2018/11/27
"""
treb_sql_string = '''
    SELECT
        rs."id",
        rs.province,
        rs.city,
        rs.address,
        rs."postalCode",
        TRIM (rs.longitude) :: NUMERIC AS longitude,
        TRIM (rs.latitude) :: NUMERIC AS latitude,
        rs.price,
        rs."buildingTypeId",
        rs."tradeTypeId",
        rs."listingDate",
        rs."ownerShipType",
        
        rs.furnished,
        rs."style",
        rs.community,
        rs."airConditioning",
        rs.washrooms,
        rs.basement1,
        rs."familyRoom",
        rs."fireplaceStove",
        rs."heatSource",
        rs."garageType",
        rs.kitchens,
        rs."parkingSpaces",
        rs."parkingIncluded",
        rs.rooms,
        rs."waterIncluded",	
        rs."totalParkingSpaces",
        rs.district,
        rs."daysOnMarket",
        rs."bedrooms",
        rs."bathroomTotal"
    FROM
        realtor_data rs
    WHERE
        1 = 1
    AND rs.latitude IS NOT NULL
    AND rs.longitude IS NOT NULL
    AND rs.latitude != ''
    AND rs.longitude != ''
    AND rs.city = 'Toronto'
'''