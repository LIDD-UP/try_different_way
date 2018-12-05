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

prediciton_query_string = '''
    SELECT 
    foo."realtorDataId" as "realtorDataId",
    foo."realtorHistoryId",
    em."id" as "estateMasterId",
    foo."mlsNumber",
    foo.price,
    foo.district,
    foo."buildingTypeId",
    foo."tradeTypeId",
    foo.latitude,
    foo.longitude,
    foo."projectDaysOnMarket",
    foo."daysOnMarket",
    foo.bedrooms,
    foo."ownerShipType",
    foo."bathroomTotal"
    from 
    (SELECT
        rs."id" as "realtorDataId",
        rh."mlsNumber",
        rh."id" as "realtorHistoryId",
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
        rs."projectDaysOnMarket",
        
        
        
        
        date_part(
            'day',
            rs."delislingDate" :: TIMESTAMP - rs."listingDate" :: TIMESTAMP
        ) AS "daysOnMarket",
        (
            rh."realtorData" -> 'Building' ->> 'Bedrooms'
        ) AS bedrooms,
        (
            rh."realtorData" -> 'Property' ->> 'OwnershipType'
        ) AS "ownerShipType",
    
            (
            rh."realtorData" -> 'Building' ->> 'BathroomTotal'
        ) AS "bathroomTotal"
    FROM
        realtor_data rs
    INNER JOIN realtor_history rh ON rs."mlsNumber" = rh."mlsNumber"
    WHERE
        1 = 1
    AND rs.latitude IS NOT NULL
    AND rs.longitude IS NOT NULL
    AND rs.latitude != ''
    AND rs.longitude != ''
    AND rs.city = 'Toronto'
    AND rs."projectDaysOnMarketFrom" = 'TREB'
    AND rs."delislingDate" IS NOT NULL
    AND rs."delislingDate" BETWEEN '2018-09-01' AND '2018-09-30') foo INNER JOIN estate_master em on foo."realtorDataId"=em."createFromSourceDataId"
    WHERE em."createFromSourceId"=1
'''