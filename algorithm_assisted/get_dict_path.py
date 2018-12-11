#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: get_dict_path.py
@time: 2018/7/17
"""

'''
the function is to get the path of dict ,
or json data path 
'''
import json


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


def getpath(d, path, ans):
    if isinstance(d, str) or isinstance(d,int) or isinstance(d,bool):
        ans.append(path[0:])
    elif isinstance(d, dict):
        for i in d:
            getpath(d[i], path+'->'+str(i), ans)
    elif isinstance(d, list):
        for i in range(len(d)):
            getpath(d[i], path+'['+str(i)+']', ans)
    return ans
realtor_json = '{"Id": "18935263", "Land": {"SizeTotal": "Unknown"}, "Building": {"Type": "House", "Bedrooms": "3", "SizeInterior": "1334 sqft", "BathroomTotal": "2"}, "Business": {}, "Distance": "", "Property": {"Type": "Single Family", "Photo": [{"LowResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/lowres/2/1730752_0.jpg", "MedResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/medres/2/1730752_0.jpg", "SequenceId": "0", "HighResPath": "https://cdn.realtor.ca/listings/TS636488467484000000/reb7/highres/2/1730752_0.jpg", "LastUpdated": "2017-12-14 11:12:28 AM"}], "Price": "$289,900", "TypeId": "300", "Address": {"Latitude": "49.496526", "Longitude": "-96.873781", "AddressText": "83 Briarwood ST|Kleefeld, Manitoba R0A0A0"}, "Parking": [{"Name": "Attached garage"}], "OwnershipType": "Freehold", "ParkingSpaceTotal": "6"}, "StatusId": "1", "MlsNumber": "1730752", "Individual": [{"Name": "Scott Kurz", "Photo": "https://cdn.realtor.ca/individuals/lowres/1285379.jpg", "Emails": [{"ContactId": "406938378"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "941-1771", "PhoneTypeId": "1"}], "LastName": "Kurz", "Websites": [{"Website": "http://scottkurz.ca", "WebsiteTypeId": "1"}, {"Website": "https://www.facebook.com/scottkurzrealestate/", "WebsiteTypeId": "2"}], "FirstName": "Scott", "IndividualID": 1944976, "Organization": {"Name": "PRESTON MYRE REAL ESTATE SERVICES WINNIPEG PROPERTIES", "Emails": [{"ContactId": "398542022"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "615-6154", "PhoneTypeId": "1"}], "Address": {"AddressText": "2 - 1000 Lorimer Boulevard|Winnipeg, MB R3P1C8"}, "HasEmail": true, "OrganizationID": 277528, "PermitFreetextEmail": true, "PermitShowListingLink": true}, "RelativeDetailsURL": "Agent/1944976/Scott-Kurz-2---1000-Lorimer-Boulevard-Winnipeg-MB-R3P1C8", "PermitFreetextEmail": true, "PermitShowListingLink": true, "CorporationDisplayTypeId": "0"}, {"Name": "Joe Fiorillo", "Emails": [{"ContactId": "398168517"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "997-7692", "PhoneTypeId": "1"}], "LastName": "Fiorillo", "FirstName": "Joe", "IndividualID": 1891043, "Organization": {"Name": "PRESTON MYRE REAL ESTATE SERVICES WINNIPEG PROPERTIES", "Emails": [{"ContactId": "398542022"}], "Phones": [{"AreaCode": "204", "PhoneType": "Telephone", "PhoneNumber": "615-6154", "PhoneTypeId": "1"}], "Address": {"AddressText": "2 - 1000 Lorimer Boulevard|Winnipeg, MB R3P1C8"}, "HasEmail": true, "OrganizationID": "277528", "PermitFreetextEmail": true, "PermitShowListingLink": true}, "RelativeDetailsURL": "Agent/1891043/Joe-Fiorillo-2---1000-Lorimer-Boulevard-Winnipeg-MB-R3P1C8", "PermitFreetextEmail": true, "PermitShowListingLink": true, "CorporationDisplayTypeId": "0"}], "PostalCode": "R0A0A0", "PublicRemarks": "R16//Kleefeld/>> PURCHASE JANUARY 24TH AND RECEIVE A ROMAN TILED SHOWER FREE! & STAINLESS STEEL APPLIANCES FREE! >> Upgraded Window Package, Upgraded Fixtures, 30+ year warrantied Laminate Flooring throughout, Maple Kitchens Cabinets with your choice of stain, soft close drawers, upper/lower crown mouldings, MASSIVE Oversized Island, 9FT Smooth Painted Ceilings Throughout, 10ft Tray Ceiling in Great Room, Larger baseboards & casings, LOADS of pot lights, full concrete driveway and walk! AND MUCH MORE! Have the confidence & peace of mind knowing that all homes are built with a *** FULL 5 YEAR NATIONAL HOME WARRANTY *** >>> BUT WAIT! ---> PURCHASE NOW and we will throw in AIR CONDITIONER UNIT FREE! WOW!â Put those finishing touches on your DREAM HOME! CALL NOW for more info and a private tour today!", "PhotoChangeDateUTC": "2017-12-14 4:12:28 PM", "RelativeDetailsURL": "/Residential/Single-Family/18935263/83-Briarwood-ST-Kleefeld-Manitoba-R0A0A0-R16"}'


realtor_dict = json.loads(realtor_json)

print(len(getpath(realtor_dict,'',[])))
path_list = getpath(realtor_dict,'',[])


full_path_list = []
for i in path_list:
    a = 'realtor_history'+str(i)
    full_path_list.append(a)

print(full_path_list)
full_path_set = set(full_path_list)
print(len(full_path_set))

with open('./path_list_first.text','w') as f:
    for i in full_path_list:
        f.writelines(i)
        f.writelines('\n')


