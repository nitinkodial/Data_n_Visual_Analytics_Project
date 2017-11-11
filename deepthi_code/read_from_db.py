import pymongo
import datetime
from dateutil.parser import parse
from pymongo import MongoClient
connection = MongoClient()
db = connection['github_database']
collection_list = db.collection_names()
start = datetime.datetime(2010, 06, 01, 0, 00, 00)
end = datetime.datetime(2010, 07, 01, 0, 00, 00)
year_user_dict = {}
universal_user = []
with open('no_of_contributors.csv', 'a+') as f:
    f.write('Term,No_of_contributors'+'\n')
    f.close()
while end.year < 2016:
    items = db.pulls.find({"created_at": {"$lt": end,"$gte": start}})
    user_list = []
    for item in items:
        if item['head'] and item['head']['user']:
            head_name = item['head']['user']['login']
        user_list.append(head_name)
        base_name = item['base']['user']['login']
        user_list.append(base_name)
    if universal_user:
        user_list = list(set(user_list) - set(universal_user))
    universal_user.extend(user_list)
    with open('no_of_contributors.csv', 'a+') as f:
        f.write(str(start.year) + '/' + str(start.month) + '/' + str(start.day) + '-' + str(end.year) + '/' + str(end.month) + '/' + str(end.day) + ',' + str(len(user_list))+ '\n') 
        f.close()
    start = start + datetime.timedelta(days=30)
    end = end + datetime.timedelta(days=30)



    
    
    
    
