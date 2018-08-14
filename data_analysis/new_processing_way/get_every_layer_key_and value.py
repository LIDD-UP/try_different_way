#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_every_layer_key_and value.py
@time: 2018/7/16
"""
import json
realtor_json = '{"Id": "18935263", "Land": {"SizeTotal": "Unknown"}, "Building": {"Type": "House", "Bedrooms": "3", "SizeInterior": "1334 sqft", "BathroomTotal": "2"}, "Business": {}, "Distance": "", "Property": {"Type": "Single Family", "Photo": [{"LowResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/lowres/2/1730752_0.jpg", "MedResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/medres/2/1730752_0.jpg", "SequenceId": "0", "HighResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/highres/2/1730752_0.jpg", "LastUpdated": "2017-12-14 11:12:28 AM"}], "Price": "$289,900", "TypeId": "300", "Address": {"Latitude": "49.496526", "Longitude": "-96.873781", "AddressText": "83 Briarwood ST|Kleefeld, Manitoba R0A0A0"}, "Parking": [{"Name": "Attached garage"}], "OwnershipType": "Freehold", "ParkingSpaceTotal": "6"}, "StatusId": "1", "MlsNumber": "1730752", "Individual": [{"Name": "Scott Kurz", "Photo": "https://cdn.realtor.ca/individuals/lowres/1285379.jpg", "Emails": [{"ContactId": "406938378"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "941-1771", "PhoneTypeId": "1"}], "LastName": "Kurz", "Websites": [{"Website": "http://scottkurz.ca", "WebsiteTypeId": "1"}, {"Website": "https://www.facebook.com/scottkurzrealestate/", "WebsiteTypeId": "2"}], "FirstName": "Scott", "IndividualID": 1944976, "Organization": {"Name": "PRESTON MYRE REAL ESTATE SERVICES WINNIPEG PROPERTIES", "Emails": [{"ContactId": "398542022"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "615-6154", "PhoneTypeId": "1"}], "Address": {"AddressText": "2 - 1000 Lorimer Boulevard|Winnipeg, MB R3P1C8"}, "HasEmail": true, "OrganizationID": 277528, "PermitFreetextEmail": true, "PermitShowListingLink": true}, "RelativeDetailsURL": "Agent/1944976/Scott-Kurz-2---1000-Lorimer-Boulevard-Winnipeg-MB-R3P1C8", "PermitFreetextEmail": true, "PermitShowListingLink": true, "CorporationDisplayTypeId": "0"}, {"Name": "Joe Fiorillo", "Emails": [{"ContactId": "398168517"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "997-7692", "PhoneTypeId": "1"}], "LastName": "Fiorillo", "FirstName": "Joe", "IndividualID": 1891043, "Organization": {"Name": "PRESTON MYRE REAL ESTATE SERVICES WINNIPEG PROPERTIES", "Emails": [{"ContactId": "398542022"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "615-6154", "PhoneTypeId": "1"}], "Address": {"AddressText": "2 - 1000 Lorimer Boulevard|Winnipeg, MB R3P1C8"}, "HasEmail": true, "OrganizationID": 277528, "PermitFreetextEmail": true, "PermitShowListingLink": true}, "RelativeDetailsURL": "Agent/1891043/Joe-Fiorillo-2---1000-Lorimer-Boulevard-Winnipeg-MB-R3P1C8", "PermitFreetextEmail": true, "PermitShowListingLink": true, "CorporationDisplayTypeId": "0"}], "PostalCode": "R0A0A0", "PublicRemarks": "R16//Kleefeld/>> PURCHASE JANUARY 24TH AND RECEIVE A ROMAN TILED SHOWER FREE! & STAINLESS STEEL APPLIANCES FREE! >> Upgraded Window Package, Upgraded Fixtures, 30+ year warrantied Laminate Flooring throughout, Maple Kitchens Cabinets with your choice of stain, soft close drawers, upper/lower crown mouldings, MASSIVE Oversized Island, 9FT Smooth Painted Ceilings Throughout, 10ft Tray Ceiling in Great Room, Larger baseboards & casings, LOADS of pot lights, full concrete driveway and walk! AND MUCH MORE! Have the confidence & peace of mind knowing that all homes are built with a *** FULL 5 YEAR NATIONAL HOME WARRANTY *** >>> BUT WAIT! ---> PURCHASE NOW and we will throw in AIR CONDITIONER UNIT FREE! WOW!â Put those finishing touches on your DREAM HOME! CALL NOW for more info and a private tour today!", "PhotoChangeDateUTC": "2017-12-14 4:12:28 PM", "RelativeDetailsURL": "/Residential/Single-Family/18935263/83-Briarwood-ST-Kleefeld-Manitoba-R0A0A0-R16"}'

realtor_dict = json.loads(realtor_json)
print(type(realtor_dict))
# print(realtor_dict)

# layers_string =''
# layers_set = set()


# # dict_value=None
# # print(dict_value_type)
#
# for k,v in realtor_dict.items():
#     dict_value_type = type(v)
#     if type(v) != dict:
#         layers_string += k
#         layers_set =layers_set.add(k)
#         print(layers_string,layers_set)
#     while(type(v) == dict):
#         first_layer = k
#         layers_string += first_layer
#         for i,j in v.items():
#             if type(v) != dict:
#                 layers_string += i

# 需要一个变量将层级连接起来：先不管层级看一下能不能将字典输出；
# def traversal_dict(dictionary,layers_string,set_layers):
#     for key, value in dictionary.items():
#         try:
#             if type(value) != dict:
#                 print(key,':',value)
#             if type(value) == dict:
#                 # layers_string += key +'->'
#                 traversal_dict(value)
#             # layers_string = ''
#         except AttributeError:
#             print("value:" +str(type(value) +'is not a dict'))
#
# layers_string =''
# set_layers =set()
# traversal_dict(realtor_dict,layers_string,set_layers)






# def dict2flatlist(d,l):
#     print(d)
#     for x in d.keys():
#         if type(d[x]) == dict:
#             dict2flatlist(d[x],l)
#         else:
#             l.append(x)
# d = {1:"a",2:"b",3:{4:"c",5:"d",6:{7:"e"}},8:"f"}
# l = []
# dict2flatlist(d,l)
# print(l)


# 将层级连接起来存入一个list里面；
def dict2flatlist(d,l,layer_string,layers):
    print(d)
    for x in d.keys():
        if type(d[x]) == dict:
            layers =len(d[x].keys())
            print(layers)
            layer_string += str(x)
            print(layer_string)
            dict2flatlist(d[x], l, layer_string,layers)

        else:
            inner_layer = str(x)
            layer_new = layer_string +inner_layer
            l.append(layer_new)
            layers = layers - 1
            print('layers',layers)

            if layers == 0:
                layer_string = ''
            print('layer_string',layer_string)


# def dict2flatlist(d,l,layer_string,layers):
#     print(d)
#     out_layer =''
#     for x in d.keys():
#         if type(d[x]) == dict:
#             # out_layer = str(x)
#             dict2flatlist(d[x], l, layer_string,layers)
#
#         else:
#             inner_layer = str(x)
#             layer_new = out_layer +inner_layer
#             l.append(layer_new)
#             print('layer_string',layer_string)
#



d = {1:"a",2:"b",3:{4:"c",5:"d",6:{7:"e"}},8:"f"}
# l = []
# layer_string = ''
# layers = 1
# dict2flatlist(d,l,layer_string,layers)
# print(l)

# def dict2flatlist(d,l,layers=0,list_key=[]):
#     print(d)
#     # count = 0
#     for x in d.keys():
#         if type(d[x]) == dict:
#             # list_key.append(str(x))
#             # layers = len(d[x].keys())
#             a = str(x)
#             # count +=1
#             layers =1
#             dict2flatlist(d[x], l,layers)
#         else:
#             list_key.append(str(x))
#             str_key = '->'.join(list_key)
#             l.append(str_key)
#             print(layers)
#
#             list_key.pop()

# def f(d):
#     ans= []
#     for i in d:
#         if isinstance(d[i],dict):
#             ans += [str(i) +j for j in f(d[i])]
#         else:
#             ans.append(str(i))
#     return ans

#
# def f(dictionary):
#     list_key = []
#     for i in dictionary:
#         if isinstance(dictionary[i],dict):
#             list_key += [str(i) + j for j in f(dictionary[i])]
#             print(f(dictionary[i]))
#             print(list_key)
#         else:
#             list_key.append(str(i))
#     return list_key

# def f(dictionary):
#     list_key = []
#     for i in dictionary:
#         if isinstance(dictionary[i],dict):
#             list_key += [str(i) + j for j in f(dictionary[i])]
#         if isinstance(dictionary[i],list):
#             list_key += [(str(i) + j for j in f(dictionary[i][k])) for k in range(len(dictionary[i]))]
#         else:
#             list_key.append(str(i))
#     return list_key
#


def getpath(d, path, ans):
    if isinstance(d, str):
        ans.append(path[2:])
    elif isinstance(d, dict):
        for i in d:
            getpath(d[i], path+'->'+str(i), ans)
    elif isinstance(d, list):
        for i in range(len(d)):
            getpath(d[i], path+'['+str(i)+']', ans)
    return ans



d={
    1: "a",
    2: "b",
    3: [{
        4: "c",
        5: "d",
        6: {7: "e"}
        },
        {8: [
            {
            9: "f",
            10: "g"
            },
            {
                11: "h",
                12: "i"
            }
            ]
        }
        ],

    13: {
        14: "j",
        15: "k"
    }
}

# 取出字典的层级关系输出 1，2，3[0]->4,3[0]->5,3[0]->6->7,3[1]->8[0]->9,3[1]->8[0]->10,3[1]->8[1]->11,3[1]->8[1]->12,13->14,13->15  还要保存到列表里面
#中括号里的数表示他在数组里的下标；


print(getpath(realtor_json,'',[]))
# 对于原数据：
# 有两种情况，一种就是一个字典，还有一种就是列表里面嵌入字典；这里还需要在获取嵌套在列表中的字典的时候获取字典的下标，

