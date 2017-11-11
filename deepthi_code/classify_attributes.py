from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn import linear_model
import matplotlib.pyplot as plt
import pymongo
import math
import datetime
import numpy as np
from dateutil.parser import parse
from pymongo import MongoClient
connection = MongoClient()
db = connection['github_database']
collection_list = db.collection_names()
start = datetime.datetime(2010, 06, 01, 0, 00, 00)
end = start + datetime.timedelta(days=45)
attributes_dict = {}
data_list = []
target_list = []

while end <= datetime.datetime(2016, 07, 01, 0, 00, 00):
    print start
    print end
    items = db.pulls.find({"created_at": {"$lt": end,"$gte": start}})
    for item in items:
        if item:
            repo_name = item['base']['repo']['name']
            if attributes_dict.get(repo_name)==None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['pulls_count'] = attributes_dict[repo_name].get('pulls_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if item['head'] and item['head']['user']:
                head_name = item['head']['user']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1
            base_name = item['base']['user']['login']
            if head_name != base_name and base_name not in attributes_dict[repo_name]['user_list']:
                attributes_dict[repo_name]['user_list'].append(base_name)
                attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1
        
    items = db.issue_comment.find({"created_at": {"$lt": end,"$gte": start}})
    for item in items:
        if item:
            repo_name = item['url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name)==None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['issue_comment_count'] = attributes_dict[repo_name].get('issue_comment_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if item['user'] and item['user']['login']:
                head_name = item['user']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1
        

    items = db.issues.find({"created_at": {"$lt": end,"$gte": start}})
    for item in items:
        if item:
            repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name)==None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['issues_count'] = attributes_dict[repo_name].get('issues_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if item['user'] and item['user']['login']:
                head_name = item['user']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    items = db.review_comments.find({"created_at": {"$lt": end,"$gte": start}})
    for item in items:
        if item:
            if item.get('repository_url'):
                repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name)==None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['review_comments_count'] = attributes_dict[repo_name].get('review_comments_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if item and item['user'] and item['user']['login']:
                head_name = item['user']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    comments = db.comments.find({"created_at": {"$lt": end,"$gte": start}})
    for comment in comments:
        if comment:
            repo_name = comment['url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name) == None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['comments_count'] = attributes_dict[repo_name].get('comments_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if comment['user']['login']:
                head_name = comment['user']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    commits = db.commits.find({"created_at": {"$lt": end,"$gte": start}})
    for commit in commits:
        if commit:
            repo_name = commit['url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name) == None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['commits_count'] = attributes_dict[repo_name].get('commits_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if commit['committer']['login']:
                head_name = commit['committer']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1
            if commit['author']['login']:
                head_name = commit['author']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    events = db.events.find({"created_at": {"$lt": end,"$gte": start}})
    for event in events:
        if event:
            repo_name = event["repo"]["url"].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name) == None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['events_count'] = attributes_dict[repo_name].get('events_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if event['actor']['login']:
                head_name = event['actor']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    issue_events = db.issue_events.find({"created_at": {"$lt": end,"$gte": start}})
    for issue_event in issue_events:
        if issue_event:
            if issue_event and issue_event['issue'] and issue_event['issue']['repository_url']:
                repo_name = issue_event["issue"]["repository_url"].split('repos/mozilla/')[1].split('/')[0]
                if attributes_dict.get(repo_name) == None:
                    attributes_dict[repo_name] = {}
                attributes_dict[repo_name]['issue_events_count'] = attributes_dict[repo_name].get('issue_events_count', 0) + 1
                if attributes_dict[repo_name].get('user_list') == None:
                    attributes_dict[repo_name]['user_list'] = []
                if issue_event['actor'] and issue_event['actor']['login']:
                    head_name = issue_event['actor']['login']
                    if head_name not in attributes_dict[repo_name]['user_list']:
                        attributes_dict[repo_name]['user_list'].append(head_name)
                        attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1

    releases = db.releases.find({"created_at": {"$lt": end,"$gte": start}})
    for release in releases:
        if release:
            repo_name = release["url"].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name) == None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['releases_count'] = attributes_dict[repo_name].get('releases_count', 0) + 1
            if attributes_dict[repo_name].get('user_list') == None:
                attributes_dict[repo_name]['user_list'] = []
            if release and release.get('author') and release['author']['login']:
                head_name = release['author']['login']
                if head_name not in attributes_dict[repo_name]['user_list']:
                    attributes_dict[repo_name]['user_list'].append(head_name)
                    attributes_dict[repo_name]['contributors_count'] = attributes_dict[repo_name].get('contributors_count', 0) + 1
        

    tags = db.tags.find({"created_at": {"$lt": end,"$gte": start}})
    for tag in tags:
        if tag:
            repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name)==None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['tags_count'] = attributes_dict[repo_name].get('tags_count', 0) + 1

    branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
    for branch in branches:
        if branch:
            repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
            if attributes_dict.get(repo_name) == None:
                attributes_dict[repo_name] = {}
            attributes_dict[repo_name]['branches_count'] = attributes_dict[repo_name].get('branches_count', 0) + 1

    start = start + datetime.timedelta(days=45)
    end = end + datetime.timedelta(days=45)

for repo_name in attributes_dict:
    data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
    target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

total_data_length = len(data_list)

train_length = int(math.floor(0.8 * len(data_list)))
data_train_list = np.array(data_list[0: train_length])
target_train_list = np.array(target_list[0: train_length])
data_test_list = np.array(data_list[train_length:])
target_test_list = np.array(target_list[train_length:])
clf = ExtraTreesClassifier()
clf = clf.fit(data_train_list, target_train_list)
print 'feature importance', clf.feature_importances_
predicted = clf.predict(data_test_list)
print 'Ensemble actual no of contributors', target_test_list
print 'Ensemble predicted no of contributors', predicted
print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted), 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted'), 'R2 score',  metrics.r2_score(target_test_list, predicted)
knn = KNeighborsClassifier()
knn = knn.fit(data_train_list, target_train_list)
predicted_knn = knn.predict(data_test_list)
print 'KNN actual no of contributors', target_test_list
print 'KNN predicted no of contributors', predicted_knn
print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted_knn), 'F1 Score: ', metrics.f1_score(target_test_list,predicted_knn, average='weighted'), 'R2 score: ', metrics.r2_score(target_test_list, predicted_knn)
clf_SGD = SGDClassifier(loss="hinge", penalty="l2")
clf_SGD = clf_SGD.fit(data_train_list, target_train_list)
predicted_SGD = clf_SGD.predict(data_test_list)
print 'Stochastic Gradient Descent prediction', predicted_SGD
print 'R2 score: ', metrics.r2_score(target_test_list, predicted_SGD)
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
y_pred_enet = enet.fit(data_train_list, target_train_list).predict(data_test_list)
print 'Elastic Net', y_pred_enet
print 'R2 score: ', metrics.r2_score(target_test_list,y_pred_enet)
lasso = Lasso(alpha=alpha)
y_pred_lasso = lasso.fit(data_train_list, target_train_list).predict(data_test_list)
print 'Lasso', y_pred_lasso
print 'R2 score: ', metrics.r2_score(target_test_list, y_pred_lasso)


        



    
    
    
    
