#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: realtor_data_source_columns.py
@time: 2018/7/16
"""

list_columns ='''
id	mlsNumber address	province	city	postalCode	price	buildingTypeId	tradeTypeId	listingDate	delislingDate	updateDate	createdTimestamp	activeFlag	updateTimestamp	contactFirstName	contactNumber	contactLastName	furnished	approxSquareFootage	trebStatus	style	municpCode	communityCode	municipalityDistrict	municipality	community	pixUpdatedDate	airConditioning	acreage	remarksForClients	aptUnit	washrooms	bedrooms	bedroomsPlus	basement1	basement2	cableTVIncluded	cacIncluded	commonElementsIncluded	frontingOn	directions	familyRoom	lotDepth	drive	utilitiesHydro	elevator	extras	farmAgriculture	fireplaceStove	lotFront	heatSource	garageType	utilitiesGas	heatType	lotIrregularities	legalDescription	lotSizeCode	kitchens	parkCost	parkingSpaces	pool	parkingIncluded	listBrokerage	room1Length	room1	room1Width	room2Length	room2	room2Width	room3Length	room3	room3Width	room4Length	room4	room4Width	room5Length	room5	room5Width	room6Length	room6	room6Width	room7Length	room7	room7Width	room8Length	room8	room8Width	room9Length	room9	room9Width	rooms	sewers	streetName	streetDirection	streetNo	streetAbbreviation	taxes	assessment	uffi	waterIncluded	washroomsType1Pcs	washroomsType2Pcs	washroomsType3Pcs	washroomsType4Pcs	washroomsType1	washroomsType2	washroomsType3	washroomsType4	waterSupplyTypes	taxYear	approxAge	zoning	typeOwnSrch	typeOwn1Out	exterior1	exterior2	otherStructures1	otherStructures2	room1Desc1	room1Desc2	room1Desc3	room2Desc1	room2Desc2	room2Desc3	room3Desc1	room3Desc2	room3Desc3	room4Desc1	room4Desc2	room4Desc3	room5Desc1	room5Desc2	room5Desc3	room6Desc1	room6Desc2	room6Desc3	room7Desc1	room7Desc2	room7Desc3	room8Desc1	room8Desc2	room8Desc3	room9Desc1	room9Desc2	room9Desc3	garageSpaces	laundryAccess	privateEntrance	kitchensPlus	laundryLevel	propertyFeatures3	propertyFeatures4	propertyFeatures5	propertyFeatures6	retirement	waterfront	specialDesignation1	specialDesignation2	areaCode	waterBodyName	waterType	waterFrontage	shorelineAllowance	shorelineExposure	parcelOfTiedLand	totalParkingSpaces	virtualTourURL	untreatedNamesPhones	411Status	411occ	district	expectedDealDate	expectedDealPrice	isDoNotCallFlag	expectedClosingDate	isDoNotMailFlag	whitePagesStatus	whitePagesOcc	addressUnit	addressStreetNumber	addressStreetName	recActiveFlag	contactInfoFrom	contactEmail	postMoverFirstName	postMoverLastName	postMoverPhoneNumber	postMoverDNCallFlag	postMoverTimestamp	processesAddress	pMlsNumber	contactInfoReferenceId	contactInfoTimestamp	postMoverReferenceId	postMoverInfoFrom	forTestUpdateTimestamp	latitude	longitude	processContactFinishFlag	waitProcessContactDate	addressProcessFlag	processAddress	

'''
list_column_split_with_space = list_columns.split('')
print(list_column_split_with_space)





feature_import = '''
address,province, city, price, 
buildingTypeId,tradeTypeId ,listingDate ,furnished,
approxSquareFootage, trebStatus ,style ,
community,airConditioning ,acreage,washrooms ,bedrooms ,
bedroomsPlus,basement1,basement2 ,cableTVIncluded,cacIncluded ,
commonElementsIncluded  ,frontingOn  ,directions,familyRoom ,lotDepth,
drive ,utilitiesHydro,extras, fireplaceStove ,lotFront, 
heatSource,garageType ,utilitiesGas ,heatType, lotIrregularities ,
kitchens, parkingSpaces,pool,parkingIncluded ,listBrokerage,
room1Length,room1,room1Width,room2Length,room2,room2Width,
room3Length,room3,room3Width,room4Length,room4,room4Width,
room5Length,room5,room5Width,room6Length,room6,room6Width,
room7Length,room7,room7Width,room8Length,room8,room8Width,
room9Length,room9,room9Width,rooms,sewers,streetName,
streetDirection,streetNo,taxes,waterIncluded,
washroomsType1Pcs,washroomsType2Pcs,washroomsType3Pcs,
washroomsType4Pcs,washroomsType1,washroomsType2,washroomsType3,
washroomsType4,taxYear,approxAge,
exterior1,exterior2,otherStructures1,garageSpaces,laundryAccess,
privateEntrance,kitchensPlus,laundryLevel,retirement,waterfront,
totalParkingSpaces,district,latitude,longitude

'''


'''
字段名称解释：
treb state： 多伦多地产局状态；
communityCode, 社区编码
municipalityDistrict ,municipality ,

parcelOfTiedLand：缺失过多，并且两种不同类别数据一个时6百多，另一个是11万，所以去掉；
'''


"province","city","address","postalCode","longitude","latitude","price","buildingTypeId","buildingTypeName","tradeTypeId","tradeTypeName","expectedDealPrice","listingDate","delislingDate","daysOnMarket"
