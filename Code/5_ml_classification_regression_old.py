from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt
import pymongo
import math
import datetime
import numpy as np
from dateutil.parser import parse
from pymongo import MongoClient
from sklearn import datasets, linear_model
import collections
from jenks import jenks


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def classify(valuesList, breaks):
  classesList = []
  for value in valuesList:
    currClass = []
    flag = 0
    
    for i in range(1, len(breaks)):
      if value < breaks[i]:
	currClass.append(i)
	flag = 1
	break
	
    if(flag==0):
      currClass.append(len(breaks) - 1)
    classesList.extend(currClass)
  
  return classesList

def main():
    
  connection = MongoClient()#connect to mongo running on host
  db = connection['github_database'] #get particular database
  collection_list = db.collection_names() #get collection names of database
  
  #resultSet = 1 #2 consecutive time slices and do regression
  resultSet = 2 #n previous n time slices and do regression
  #resultSet = 3 #n previous n time slices and do classification
  #resultSet = 4 #2 consecutive time slices and do classification
  
  start_const = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
  final_end_const = datetime.datetime(2016, 7, 1, 0, 00, 00)
  delta_const = datetime.timedelta(days=30)#45)
  

  ##Code for results set 1
  '''
  start = datetime.datetime(2010, 06, 01, 0, 00, 00)
  end = start + datetime.timedelta(days=45)
  attributes_dict = {}
  data_list = []
  target_list = []
  varI = 0
  while end <= datetime.datetime(2016, 7, 1, 0, 00, 00):
      print start
      print end
      #print db
      print varI
      varI = varI + 1
      start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
      end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
      
      #items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
      items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
      #print "debug 0.1"
      ##print(db.collection_names())  
      for item in items:
	  #print "debug 0"
	  if item:
	      repo_name = item['base']['repo']['name']
	      #print item['created_at']
	      #wait = input("PRESS ENTER TO CONTINUE.")
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
	  #print "debug 1"
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
	  #print "debug 2"
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
	  #print "debug 3"
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
	  #print "debug 4"
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
	  #print "debug 5"
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
	  #print "debug 6"
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
	  #print "debug 7"
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
	  #print "debug 8"
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
	  #print "debug 9"
	  if tag:
	      repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
	      if attributes_dict.get(repo_name)==None:
		  attributes_dict[repo_name] = {}
	      attributes_dict[repo_name]['tags_count'] = attributes_dict[repo_name].get('tags_count', 0) + 1

      branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
      for branch in branches:
	  #print "debug 10"
	  if branch:
	      repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
	      if attributes_dict.get(repo_name) == None:
		  attributes_dict[repo_name] = {}
	      attributes_dict[repo_name]['branches_count'] = attributes_dict[repo_name].get('branches_count', 0) + 1

      start = start + datetime.timedelta(days=45)
      end = end + datetime.timedelta(days=45)

  for repo_name in attributes_dict:
      print repo_name
      data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
      target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

  total_data_length = len(data_list)
  print total_data_length
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

  '''

  '''       
  ##Code for results set 2
  print "results set 2"
  start = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
  end = start + datetime.timedelta(days=45)
  attributes_dict = {}
  data_list = []
  target_list = []
  varI = 0
  while end <= datetime.datetime(2016, 7, 1, 0, 00, 00):
      print start
      print end
      #print db
      print varI
      varI = varI + 1
      start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
      end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
      
      #items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
      #print "debug 0.1"
      ##print(db.collection_names())  
      
      items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
      for item in items:
	  #print "debug 0"
	  if item:
	      repo_name = item['base']['repo']['name']
	      #print item['created_at']
	      #wait = input("PRESS ENTER TO CONTINUE.")
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
	  #print "debug 1"
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
	  #print "debug 2"
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
	  #print "debug 3"
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
	  #print "debug 4"
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
	  #print "debug 5"
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
	  #print "debug 6"
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
	  #print "debug 7"
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
	  #print "debug 8"
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
	  #print "debug 9"
	  if tag:
	      repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
	      if attributes_dict.get(repo_name)==None:
		  attributes_dict[repo_name] = {}
	      attributes_dict[repo_name]['tags_count'] = attributes_dict[repo_name].get('tags_count', 0) + 1

      branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
      for branch in branches:
	  #print "debug 10"
	  if branch:
	      repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
	      if attributes_dict.get(repo_name) == None:
		  attributes_dict[repo_name] = {}
	      attributes_dict[repo_name]['branches_count'] = attributes_dict[repo_name].get('branches_count', 0) + 1

      start = start + datetime.timedelta(days=45)
      end = end + datetime.timedelta(days=45)

  for repo_name in attributes_dict:
      print repo_name
      data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
      #data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
      #data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
      target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

  total_data_length = len(data_list)
  print total_data_length
  train_length = int(math.floor(0.8 * len(data_list)))
  data_train_list = np.array(data_list[0: train_length])
  target_train_list = np.array(target_list[0: train_length])
  data_test_list = np.array(data_list[train_length:])
  target_test_list = np.array(target_list[train_length:])

  # start of Linear Regression
  print "\nLinear Regression"
  regr = linear_model.LinearRegression()
  regr.fit(data_train_list, target_train_list)

  # The coefficients
  print('Coefficients: \n', regr.coef_)
  # The mean squared error
  print("Mean squared error: %.2f"
	% np.mean((regr.predict(data_test_list) - target_test_list) ** 2))
  # Explained variance score: 1 is perfect prediction
  print('Variance score: %.2f' % regr.score(data_test_list, target_test_list))
  #print data_test_list.shape[0]
  #for varI in range(data_test_list.shape[0]):
    #print "Prediction: %.2f, GroundTruth: %.2f" % (regr.predict(data_test_list[varI]), target_test_list[varI])

  # start of Decision tree Regression
  print "\nDecision tree Regression"
  model = tree.DecisionTreeRegressor() #for regression
  # Train the model using the training sets and check score
  model.fit(data_train_list, target_train_list)
  model.score(data_train_list, target_train_list)
  #Predict Output
  predicted = model.predict(data_test_list)
  print("Mean squared error: %.2f"
	% np.mean((model.predict(data_test_list) - target_test_list) ** 2))
  print('Variance score: %.2f\n' % model.score(data_test_list, target_test_list))
  #for varI in range(data_test_list.shape[0]):
    #print "Prediction: %.2f, GroundTruth: %.2f" % (model.predict(data_test_list[varI]), target_test_list[varI])

  # Plot outputs
  #plt.scatter(data_test_list, target_test_list,  color='black')
  #plt.plot(data_test_list, regr.predict(data_test_list), color='blue',
	  #linewidth=3)

  #plt.xticks(())
  #plt.yticks(())

  #plt.show()

  #wait = input("PRESS ENTER TO CONTINUE.")

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


  '''

      
  if(resultSet == 1):
    ##Code for results set 1 # runs regression using prev time slice data and prev contri count and predicts current count
    #'''
    #%%
    
    print "Code for results set 1"
    #start_const = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
    #final_end_const = datetime.datetime(2016, 7, 1, 0, 00, 00)
    #delta_const = datetime.timedelta(days=45)
    start = start_const
    end = start + delta_const
    attributes_dict = {}
    data_list = []
    target_list = []
    target_list_dates = []
    varI = 0
    while end <= final_end_const:
	print start
	print end
	#print db
	print varI
	varI = varI + 1
	start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
	end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
	
	#items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
	#print "debug 0.1"
	##print(db.collection_names())  
	attributes_dict[start] = {}
	#print attributes_dict
	items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
	for item in items:
	    #print "debug 0"
	    if item:
		repo_name = item['base']['repo']['name']
		#print item['created_at']
		#wait = input("PRESS ENTER TO CONTINUE.")
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['pulls_count'] = attributes_dict[start][repo_name].get('pulls_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['head'] and item['head']['user']:
		    head_name = item['head']['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		base_name = item['base']['user']['login']
		if head_name != base_name and base_name not in attributes_dict[start][repo_name]['user_list']:
		    attributes_dict[start][repo_name]['user_list'].append(base_name)
		    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    
	items = db.issue_comment.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 1"
	    if item:
		repo_name = item['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issue_comment_count'] = attributes_dict[start][repo_name].get('issue_comment_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	items = db.issues.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 2"
	    if item:
		repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issues_count'] = attributes_dict[start][repo_name].get('issues_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	items = db.review_comments.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 3"
	    if item:
		if item.get('repository_url'):
		    repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['review_comments_count'] = attributes_dict[start][repo_name].get('review_comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item and item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	comments = db.comments.find({"created_at": {"$lt": end,"$gte": start}})
	for comment in comments:
	    #print "debug 4"
	    if comment:
		repo_name = comment['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['comments_count'] = attributes_dict[start][repo_name].get('comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if comment['user']['login']:
		    head_name = comment['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	commits = db.commits.find({"created_at": {"$lt": end,"$gte": start}})
	for commit in commits:
	    #print "debug 5"
	    if commit:
		repo_name = commit['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['commits_count'] = attributes_dict[start][repo_name].get('commits_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if commit['committer']['login']:
		    head_name = commit['committer']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		if commit['author']['login']:
		    head_name = commit['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	events = db.events.find({"created_at": {"$lt": end,"$gte": start}})
	for event in events:
	    #print "debug 6"
	    if event:
		repo_name = event["repo"]["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['events_count'] = attributes_dict[start][repo_name].get('events_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if event['actor']['login']:
		    head_name = event['actor']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	issue_events = db.issue_events.find({"created_at": {"$lt": end,"$gte": start}})
	for issue_event in issue_events:
	    #print "debug 7"
	    if issue_event:
		if issue_event and issue_event['issue'] and issue_event['issue']['repository_url']:
		    repo_name = issue_event["issue"]["repository_url"].split('repos/mozilla/')[1].split('/')[0]
		    if attributes_dict[start].get(repo_name) == None:
			attributes_dict[start][repo_name] = {}
		    attributes_dict[start][repo_name]['issue_events_count'] = attributes_dict[start][repo_name].get('issue_events_count', 0) + 1
		    if attributes_dict[start][repo_name].get('user_list') == None:
			attributes_dict[start][repo_name]['user_list'] = []
		    if issue_event['actor'] and issue_event['actor']['login']:
			head_name = issue_event['actor']['login']
			if head_name not in attributes_dict[start][repo_name]['user_list']:
			    attributes_dict[start][repo_name]['user_list'].append(head_name)
			    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	releases = db.releases.find({"created_at": {"$lt": end,"$gte": start}})
	for release in releases:
	    #print "debug 8"
	    if release:
		repo_name = release["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['releases_count'] = attributes_dict[start][repo_name].get('releases_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if release and release.get('author') and release['author']['login']:
		    head_name = release['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	tags = db.tags.find({"created_at": {"$lt": end,"$gte": start}})
	for tag in tags:
	    #print "debug 9"
	    if tag:
		repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['tags_count'] = attributes_dict[start][repo_name].get('tags_count', 0) + 1

	branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
	for branch in branches:
	    #print "debug 10"
	    if branch:
		repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['branches_count'] = attributes_dict[start][repo_name].get('branches_count', 0) + 1

	start = start + delta_const
	end = end + delta_const


    #%%    
    prev = start_const
    start = start_const + delta_const
    end = start + delta_const
    varI = 0
    #print attributes_dict#.values()
    while end <= final_end_const:
	#print 'nitk'
	#print prev
	#print start
	#print end
	#print db
	#print varI
	varI = varI + 1
	curr_date_repos =  attributes_dict[start] #dictionary of repos for a given date
	prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date
	#print curr_date_repos
	#print attributes_dict[start]
	
	curr_time_pulls_count = 0
	curr_time_issue_comment_count = 0
	curr_time_issues_count = 0
	curr_time_review_comments_count = 0
	curr_time_comments_count = 0

	curr_time_commits_count = 0
	curr_time_events_count = 0
	curr_time_issue_events_count = 0
	curr_time_releases_count = 0
	curr_time_tags_count = 0

	curr_time_contributors_count = 0
	curr_time_future_contributors_count_actual = 0
	

	
	for item in curr_date_repos: #iterate over keys, i.e. repos here
	  #print item
	  attributes_dict[start][item]['prev_contributors_count'] = attributes_dict[start][item].get('prev_contributors_count', 0)
	  #print "debug 1"
	  #print attributes_dict[start][item]
	  if item in prev_date_repos:
	    attributes_dict[start][item]['prev_contributors_count'] = attributes_dict[prev][item]['contributors_count']
	    data_list.append([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
			    attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
			    attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
			    attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
			    attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
			    attributes_dict[prev][item].get('contributors_count', 0)])
	    target_list.append(attributes_dict[start][item].get('contributors_count', 0))
	    target_list_dates.append(prev)
	    
	    print "Data for presentation:;{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(prev, item,
	    attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
		      attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
		      attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
		      attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
		      attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
		      attributes_dict[prev][item].get('contributors_count', 0),attributes_dict[start][item].get('contributors_count', 0))
	    
	    curr_time_pulls_count = curr_time_pulls_count + attributes_dict[prev][item].get('pulls_count', 0)
	    curr_time_issue_comment_count = curr_time_issue_comment_count + attributes_dict[prev][item].get('issue_comment_count', 0)
	    curr_time_issues_count = curr_time_issues_count + attributes_dict[prev][item].get('issues_count', 0)
	    curr_time_review_comments_count = curr_time_review_comments_count + attributes_dict[prev][item].get('review_comments_count', 0)
	    curr_time_comments_count = curr_time_comments_count + attributes_dict[prev][item].get('comments_count', 0)

	    curr_time_commits_count = curr_time_commits_count + attributes_dict[prev][item].get('commits_count', 0)
	    curr_time_events_count = curr_time_events_count + attributes_dict[prev][item].get('events_count', 0)
	    curr_time_issue_events_count = curr_time_issue_events_count + attributes_dict[prev][item].get('issue_events_count', 0)
	    curr_time_releases_count = curr_time_releases_count + attributes_dict[prev][item].get('releases_count', 0)
	    curr_time_tags_count = curr_time_tags_count + attributes_dict[prev][item].get('tags_count', 0)

	    curr_time_contributors_count = curr_time_contributors_count + attributes_dict[prev][item].get('contributors_count', 0)
	    curr_time_future_contributors_count_actual = curr_time_future_contributors_count_actual + attributes_dict[start][item].get('contributors_count', 0)
	    


	    #print "debug 2"
	    #print attributes_dict[prev][item]
	  #print "debug 3"	  
	  #print attributes_dict[start][item]
	
	#print "Data for presentation2:;{};{};{};{};{};{};{};{};{};{};{};{};{}".format(prev, 
	    #curr_time_pulls_count, curr_time_issue_comment_count,
	    #curr_time_issues_count, curr_time_review_comments_count,
	    #curr_time_comments_count, curr_time_commits_count,
	    #curr_time_events_count, curr_time_issue_events_count,
	    #curr_time_releases_count, curr_time_tags_count, 
	    #curr_time_contributors_count, curr_time_future_contributors_count_actual)
	    
	    
	  
	prev = prev + delta_const
	start = start + delta_const
	end = end + delta_const    

    print data_list 
    print target_list
    #%%

    print data_list 
    print target_list
    #for repo_name in attributes_dict:
	#print repo_name
	#data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	#target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

    total_data_length = len(data_list)
    print total_data_length
    train_length = int(math.floor(0.8 * len(data_list)))
    data_train_list = np.array(data_list[0: train_length])
    target_train_list = np.array(target_list[0: train_length])
    target_train_list_dates = np.array(target_list_dates[0: train_length])
    data_test_list = np.array(data_list[train_length:])
    target_test_list = np.array(target_list[train_length:])
    target_test_list_dates = np.array(target_list_dates[train_length:])

    print "data_train_list rows",data_train_list.shape[0]
    print "data_test_list rows",data_test_list.shape[0]
    #print "target_test_list",target_test_list
        
    #for varI in range(data_test_list.shape[0]):
      #print "output data:;{};{}".format(data_test_list[varI,:], target_list[varI])
    # start of Linear Regression
    print "\nLinear Regression"
    regr = linear_model.LinearRegression()
    regr.fit(data_train_list, target_train_list)
    predicted = regr.predict(data_test_list)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
	  % np.mean((regr.predict(data_test_list) - target_test_list) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(data_test_list, target_test_list))
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted)
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')

    #print data_test_list.shape[0]
    #for varI in range(data_test_list.shape[0]):
      #print "Prediction: %.2f, GroundTruth: %.2f" % (regr.predict(data_test_list[varI]), target_test_list[varI])

    # start of Decision tree Regression
    print "\nDecision tree Regression"
    model = tree.DecisionTreeRegressor() #for regression
    # Train the model using the training sets and check score
    model.fit(data_train_list, target_train_list)
    model.score(data_train_list, target_train_list)
    #Predict Output
    predicted = model.predict(data_test_list)
    print("Mean squared error: %.2f"
	  % np.mean((model.predict(data_test_list) - target_test_list) ** 2))
    print('Variance score: %.2f\n' % model.score(data_test_list, target_test_list))
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted),
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')

    #for varI in range(data_test_list.shape[0]):
      #print "Prediction: %.2f, GroundTruth: %.2f" % (model.predict(data_test_list[varI]), target_test_list[varI])

    # Plot outputs
    #plt.scatter(data_test_list, target_test_list,  color='black')
    #plt.plot(data_test_list, regr.predict(data_test_list), color='blue',
	    #linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()

    #wait = input("PRESS ENTER TO CONTINUE.")    
    
    clf = ExtraTreesRegressor()#ExtraTreesClassifier()
    clf = clf.fit(data_train_list, target_train_list)
    print 'feature importance', clf.feature_importances_
    predicted = clf.predict(data_test_list)
    print 'ExtraTreesRegressor Ensemble actual no of contributors', target_test_list
    print 'ExtraTreesRegressor Ensemble predicted no of contributors', predicted
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted)
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')

    knn = KNeighborsRegressor()#KNeighborsClassifier()
    knn = knn.fit(data_train_list, target_train_list)
    predicted_knn = knn.predict(data_test_list)
    print 'KNN actual no of contributors', target_test_list
    print 'KNN predicted no of contributors', predicted_knn
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted_knn), 
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted_knn, average='weighted')
    print 'R2 score: ', metrics.r2_score(target_test_list, predicted_knn)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted_knn, sample_weight=None, multioutput='uniform_average')
    
    clf_SGD = SGDRegressor(loss="squared_loss", penalty="l2")
    clf_SGD = clf_SGD.fit(data_train_list, target_train_list)
    predicted_SGD = clf_SGD.predict(data_test_list)
    print 'Stochastic Gradient Descent prediction', predicted_SGD
    print 'R2 score: ', metrics.r2_score(target_test_list, predicted_SGD)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted_SGD, sample_weight=None, multioutput='uniform_average')
    
    alpha = 0.1
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
    y_pred_enet = enet.fit(data_train_list, target_train_list).predict(data_test_list)
    print 'Elastic Net', y_pred_enet
    print 'R2 score: ', metrics.r2_score(target_test_list,y_pred_enet)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, y_pred_enet, sample_weight=None, multioutput='uniform_average')
    
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(data_train_list, target_train_list).predict(data_test_list)
    print 'Lasso', y_pred_lasso
    print 'R2 score: ', metrics.r2_score(target_test_list, y_pred_lasso)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, y_pred_lasso, sample_weight=None, multioutput='uniform_average')
    
    
    for varI in range(data_test_list.shape[0]):
      print "output data:;{};{};{};{}".format(target_test_list_dates[varI], data_test_list[varI,:], target_test_list[varI], y_pred_lasso[varI] )
      
    prev_date = target_test_list_dates[0]
    target_test_list_date_grouped = []
    y_pred_lasso_date_grouped = []
    
    target_test_list_date_tmp = 0
    y_pred_lasso_date_tmp = 0
    
    print "\n\n\n"
    for varI in range(data_test_list.shape[0]):
      #print "output data:;{};{};{};{}".format(target_test_list_dates[varI], data_test_list[varI,:], target_test_list[varI], y_pred_lasso[varI] )
      if(varI == data_test_list.shape[0]-1):
	target_test_list_date_tmp = target_test_list_date_tmp + target_test_list[varI]
	y_pred_lasso_date_tmp = y_pred_lasso_date_tmp + y_pred_lasso[varI]
	print "output data:;{};{};{}".format(prev_date,target_test_list_date_tmp,y_pred_lasso_date_tmp)
	
      if(target_test_list_dates[varI]==prev_date):
	target_test_list_date_tmp = target_test_list_date_tmp + target_test_list[varI]
	y_pred_lasso_date_tmp = y_pred_lasso_date_tmp + y_pred_lasso[varI]
      else:	
	target_test_list_date_grouped.append(target_test_list_date_tmp)
	y_pred_lasso_date_grouped.append(y_pred_lasso_date_tmp)
	print "output data:;{};{};{}".format(prev_date,target_test_list_date_tmp,y_pred_lasso_date_tmp)
	prev_date = target_test_list_dates[varI]
	target_test_list_date_tmp = target_test_list[varI]
	y_pred_lasso_date_tmp = y_pred_lasso[varI]
	
    #print varI
	

    #'''


      
      
      









  elif(resultSet == 2):    
    ##Code for results set 2 #regression using 'n' time slices
    #'''
    #%%
    print "Code for results set 2"
    ##start_const = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
    ##final_end_const = datetime.datetime(2016, 7, 1, 0, 00, 00)
    ##delta_const = datetime.timedelta(days=45)#45
    start = start_const
    end = start + delta_const
    attributes_dict = {}
    data_list = []
    target_list = []
    varI = 0
    while end <= final_end_const:
	print start
	print end
	#print db
	print varI
	varI = varI + 1
	start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
	end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
	
	#items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
	#print "debug 0.1"
	##print(db.collection_names())  
	attributes_dict[start] = {}
	#print attributes_dict
	items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
	for item in items:
	    #print "debug 0"        
	    if item:
		repo_name = item['base']['repo']['name']
		#print item['created_at']
		#wait = input("PRESS ENTER TO CONTINUE.")
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['pulls_count'] = attributes_dict[start][repo_name].get('pulls_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['head'] and item['head']['user']:
		    head_name = item['head']['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		base_name = item['base']['user']['login']
		if head_name != base_name and base_name not in attributes_dict[start][repo_name]['user_list']:
		    attributes_dict[start][repo_name]['user_list'].append(base_name)
		    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    
	items = db.issue_comment.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 1"
	    if item:
		repo_name = item['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issue_comment_count'] = attributes_dict[start][repo_name].get('issue_comment_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	items = db.issues.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 2"
	    if item:
		repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issues_count'] = attributes_dict[start][repo_name].get('issues_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	items = db.review_comments.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 3"
	    if item:
		if item.get('repository_url'):
		    repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['review_comments_count'] = attributes_dict[start][repo_name].get('review_comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item and item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	comments = db.comments.find({"created_at": {"$lt": end,"$gte": start}})
	for comment in comments:
	    #print "debug 4"
	    if comment:
		repo_name = comment['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['comments_count'] = attributes_dict[start][repo_name].get('comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if comment['user']['login']:
		    head_name = comment['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	commits = db.commits.find({"created_at": {"$lt": end,"$gte": start}})
	for commit in commits:
	    #print "debug 5"
	    if commit:
		repo_name = commit['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['commits_count'] = attributes_dict[start][repo_name].get('commits_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if commit['committer']['login']:
		    head_name = commit['committer']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		if commit['author']['login']:
		    head_name = commit['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	events = db.events.find({"created_at": {"$lt": end,"$gte": start}})
	for event in events:
	    #print "debug 6"
	    if event:
		repo_name = event["repo"]["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['events_count'] = attributes_dict[start][repo_name].get('events_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if event['actor']['login']:
		    head_name = event['actor']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	issue_events = db.issue_events.find({"created_at": {"$lt": end,"$gte": start}})
	for issue_event in issue_events:
	    #print "debug 7"
	    if issue_event:
		if issue_event and issue_event['issue'] and issue_event['issue']['repository_url']:
		    repo_name = issue_event["issue"]["repository_url"].split('repos/mozilla/')[1].split('/')[0]
		    if attributes_dict[start].get(repo_name) == None:
			attributes_dict[start][repo_name] = {}
		    attributes_dict[start][repo_name]['issue_events_count'] = attributes_dict[start][repo_name].get('issue_events_count', 0) + 1
		    if attributes_dict[start][repo_name].get('user_list') == None:
			attributes_dict[start][repo_name]['user_list'] = []
		    if issue_event['actor'] and issue_event['actor']['login']:
			head_name = issue_event['actor']['login']
			if head_name not in attributes_dict[start][repo_name]['user_list']:
			    attributes_dict[start][repo_name]['user_list'].append(head_name)
			    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	releases = db.releases.find({"created_at": {"$lt": end,"$gte": start}})
	for release in releases:
	    #print "debug 8"
	    if release:
		repo_name = release["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['releases_count'] = attributes_dict[start][repo_name].get('releases_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if release and release.get('author') and release['author']['login']:
		    head_name = release['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	tags = db.tags.find({"created_at": {"$lt": end,"$gte": start}})
	for tag in tags:
	    #print "debug 9"
	    if tag:
		repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['tags_count'] = attributes_dict[start][repo_name].get('tags_count', 0) + 1

	branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
	for branch in branches:
	    #print "debug 10"
	    if branch:
		repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['branches_count'] = attributes_dict[start][repo_name].get('branches_count', 0) + 1

	start = start + delta_const
	end = end + delta_const

    #print "attributes_dict\n",attributes_dict
    #%%    
    num_time_slices = 12 #12 is best results #this will give num_time_slices+1
    num_time_slices = num_time_slices - 1 #this will give original num_time_slices (e.g. no of days from 1 to 3 is 3)
    start = start_const + num_time_slices*delta_const #start after num_time_slices of delta
    prev = start - num_time_slices*delta_const  #start before num_time_slices of delta
    end = start + delta_const
    varI = 0
    #print "nitk1",prev,start
    #print attributes_dict#.values()
    max_num_repos = 0
    while end <= final_end_const:
	
	prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date    
	  
	start_tmp = start #this is used in case there are time slices in which some repos are missing, so start_tmp will increment in those cases
	curr_max_num_repos = 0
	for item in prev_date_repos:#every entry in data will be for a particular repo. item is a repo
	  #print "varI",varI
	  
	  varI = varI + 1
	  tmp_data_list = [] #store N-1 time slices of data (attributes) for that repo
	  tmp_target_val = []
	  num_time_slices_obtained = 0 #for each item
	  while(prev<start_tmp):#while loop to get previous n time slices and to add it as features
	    prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date 
	    #print item
	    #print "prev, start_tmp ", prev, start_tmp
	    #print prev_date_repos
	    
	    dict_found_flag = 1
	    
	    while(item not in attributes_dict[prev]):
	      #print item
	      prev = prev + delta_const #skip that date
	      start_tmp = start_tmp + delta_const #increment upper limit as well,since a time slice was skipped
	      dict_found_flag = 0
	      if(prev in attributes_dict):
		if(item in attributes_dict[prev]):#if item is finally found in some date
		  dict_found_flag = 1
	      else:
		break
	    
	    if(dict_found_flag==1):  
	      tmp_data_list.extend([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
				  attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
				  attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
				  attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
				  attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
				  attributes_dict[prev][item].get('contributors_count', 0)])
	      num_time_slices_obtained = num_time_slices_obtained + 1
	      #print "xx",num_time_slices_obtained
			      
	    prev = prev + delta_const
	    if(prev>(final_end_const-delta_const)):#if no time slices and is about to go out of bounds
	      break
	    
	  #end of while(prev<start_tmp)
	  
	  #print item
	  #print prev,start
	  #print prev_date_repos
	  #at this point after exiting while loop, prev = start
	  #prev_date_repos =  attributes_dict[prev] #when prev = start, then don't add 'contributors_count' as feature as that is output to be predicted     
	  dict_found_flag = 1
	  if(prev<=(final_end_const-delta_const)):
	    while(item not in attributes_dict[prev]):
	      prev = prev + delta_const #skip that date
	      start_tmp = start_tmp + delta_const #increment upper limit as well,since a time slice was skipped
	      dict_found_flag = 0
	      if(prev in attributes_dict):
		if(item in attributes_dict[prev]):#if item is finally found in some date
		  dict_found_flag = 1
	      else:
		#print "debug 0.1"
		break
	      
	    if(dict_found_flag==1):   
	      tmp_target_val =([attributes_dict[prev][item].get('contributors_count', 0)])
	      #tmp_data_list.extend([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
				  #attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
				  #attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
				  #attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
				  #attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0)])
	      
	      #print "debug 0.2",attributes_dict[prev][item].get('contributors_count', 0)
	      num_time_slices_obtained = num_time_slices_obtained + 1
	      #print "debug 0.2",num_time_slices_obtained
	  #print "debug 0.0",prev
	  #print "num_time_slices_obtained",num_time_slices_obtained
	  if(num_time_slices_obtained == (num_time_slices+1)): #only if 6 (or n) time slices were obtained, then only add to ML data
	    data_list.append(tmp_data_list)	
	    target_list.append(tmp_target_val)

	    #print "debug 1"
	    #target_list.append(attributes_dict[prev+delta_const][item].get('contributors_count', 0))
	    curr_max_num_repos = curr_max_num_repos + 1
	    #print "debug 2"
	  prev = start - num_time_slices*delta_const  #start before num_time_slices of delta, reset prev for next iteration of for loop
	  start_tmp = start
	  #print "debug 3",prev
	  #print "debug 4",start
	#end of for item in prev_date_repos
	
	if(curr_max_num_repos>max_num_repos):
	  max_num_repos = curr_max_num_repos
	  
	start = start + delta_const
	prev = start - num_time_slices*delta_const  #start before num_time_slices of delta
	end = end + delta_const    

    #print data_list 
    #print target_list
    #%%
    print "max num of repos processed in 'n'* delta time slices", max_num_repos
    print data_list 
    print target_list
    #for repo_name in attributes_dict:
	#print repo_name
	#data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	#target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

    total_data_length = len(data_list)
    print total_data_length
    train_length = int(math.floor(0.8 * len(data_list)))
    data_train_list = np.array(data_list[0: train_length])
    target_train_list = np.array(target_list[0: train_length])
    data_test_list = np.array(data_list[train_length:])
    target_test_list = np.array(target_list[train_length:])
    
    
    
    
    
    #for varI in range(data_test_list.shape[0]):
      #print "output data:;{};{}".format(data_test_list[varI,:], target_list[varI])
    # start of Linear Regression
    print "\nLinear Regression"
    regr = linear_model.LinearRegression()
    regr.fit(data_train_list, target_train_list)
    predicted = regr.predict(data_test_list)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
	  % np.mean((regr.predict(data_test_list) - target_test_list) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(data_test_list, target_test_list))
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted)
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')

    #print data_test_list.shape[0]
    #for varI in range(data_test_list.shape[0]):
      #print "Prediction: %.2f, GroundTruth: %.2f" % (regr.predict(data_test_list[varI]), target_test_list[varI])

    # start of Decision tree Regression
    print "\nDecision tree Regression"
    model = tree.DecisionTreeRegressor() #for regression
    # Train the model using the training sets and check score
    model.fit(data_train_list, target_train_list)
    model.score(data_train_list, target_train_list)
    #Predict Output
    predicted = model.predict(data_test_list)
    print("Mean squared error: %.2f"
	  % np.mean((model.predict(data_test_list) - target_test_list) ** 2))
    print('Variance score: %.2f\n' % model.score(data_test_list, target_test_list))
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted),
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')
    #for varI in range(data_test_list.shape[0]):
      #print "Prediction: %.2f, GroundTruth: %.2f" % (model.predict(data_test_list[varI]), target_test_list[varI])

    # Plot outputs
    #plt.scatter(data_test_list, target_test_list,  color='black')
    #plt.plot(data_test_list, regr.predict(data_test_list), color='blue',
	    #linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()

    #wait = input("PRESS ENTER TO CONTINUE.")    
    
    clf = ExtraTreesRegressor()#ExtraTreesClassifier()
    clf = clf.fit(data_train_list, target_train_list)
    print 'feature importance', clf.feature_importances_
    predicted = clf.predict(data_test_list)
    print 'ExtraTreesRegressor Ensemble actual no of contributors', target_test_list
    print 'ExtraTreesRegressor Ensemble predicted no of contributors', predicted
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted)
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted, average='weighted')
    print 'R2 score',  metrics.r2_score(target_test_list, predicted)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted, sample_weight=None, multioutput='uniform_average')
    
    knn = KNeighborsRegressor()#KNeighborsClassifier()
    knn = knn.fit(data_train_list, target_train_list)
    predicted_knn = knn.predict(data_test_list)
    print 'KNN actual no of contributors', target_test_list
    print 'KNN predicted no of contributors', predicted_knn
    #print 'Accuracy: ', metrics.accuracy_score(target_test_list, predicted_knn), 
    #print 'F1 Score: ', metrics.f1_score(target_test_list,predicted_knn, average='weighted')
    print 'R2 score: ', metrics.r2_score(target_test_list, predicted_knn)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted_knn, sample_weight=None, multioutput='uniform_average')
    
    clf_SGD = SGDRegressor(loss="squared_loss", penalty="l2")
    clf_SGD = clf_SGD.fit(data_train_list, target_train_list)
    predicted_SGD = clf_SGD.predict(data_test_list)
    print 'Stochastic Gradient Descent prediction', predicted_SGD
    print 'R2 score: ', metrics.r2_score(target_test_list, predicted_SGD)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, predicted_SGD, sample_weight=None, multioutput='uniform_average')
    
    alpha = 0.1
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
    y_pred_enet = enet.fit(data_train_list, target_train_list).predict(data_test_list)
    print 'Elastic Net', y_pred_enet
    print 'R2 score: ', metrics.r2_score(target_test_list,y_pred_enet)
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, y_pred_enet, sample_weight=None, multioutput='uniform_average')

    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(data_train_list, target_train_list).predict(data_test_list)
    print 'Lasso', y_pred_lasso
    print 'R2 score: ', metrics.r2_score(target_test_list, y_pred_lasso)            
    print 'Mean squared error: ', metrics.mean_squared_error(target_test_list, y_pred_lasso, sample_weight=None, multioutput='uniform_average')

    #'''



  elif(resultSet == 3):
    ##Code for results set 3
    ##classification using 'n' time slices of data 
    #'''
    #%%
    
    print "Code for results set 3"
    ##start_const = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
    ##final_end_const = datetime.datetime(2016, 7, 1, 0, 00, 00)#(2012, 7, 1, 0, 00, 00)#(2016, 7, 1, 0, 00, 00)
    ##delta_const = datetime.timedelta(days=45)#45
    start = start_const
    end = start + delta_const
    attributes_dict = {}
    data_list = []
    target_list = []
    list_of_all_Y_vals = []
    data_target_list_dict = {}
    varI = 0
    while end <= final_end_const:
	print start
	print end
	#print db
	print varI
	varI = varI + 1
	start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
	end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
	
	#items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
	#print "debug 0.1"
	##print(db.collection_names())  
	attributes_dict[start] = {}
	#print attributes_dict
	items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
	for item in items:
	    #print "debug 0"        
	    if item:
		repo_name = item['base']['repo']['name']
		#print item['created_at']
		#wait = input("PRESS ENTER TO CONTINUE.")
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['pulls_count'] = attributes_dict[start][repo_name].get('pulls_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['head'] and item['head']['user']:
		    head_name = item['head']['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		base_name = item['base']['user']['login']
		if head_name != base_name and base_name not in attributes_dict[start][repo_name]['user_list']:
		    attributes_dict[start][repo_name]['user_list'].append(base_name)
		    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    
	items = db.issue_comment.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 1"
	    if item:
		repo_name = item['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issue_comment_count'] = attributes_dict[start][repo_name].get('issue_comment_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	items = db.issues.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 2"
	    if item:
		repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issues_count'] = attributes_dict[start][repo_name].get('issues_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	items = db.review_comments.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 3"
	    if item:
		if item.get('repository_url'):
		    repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['review_comments_count'] = attributes_dict[start][repo_name].get('review_comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item and item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	comments = db.comments.find({"created_at": {"$lt": end,"$gte": start}})
	for comment in comments:
	    #print "debug 4"
	    if comment:
		repo_name = comment['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['comments_count'] = attributes_dict[start][repo_name].get('comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if comment['user']['login']:
		    head_name = comment['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	commits = db.commits.find({"created_at": {"$lt": end,"$gte": start}})
	for commit in commits:
	    #print "debug 5"
	    if commit:
		repo_name = commit['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['commits_count'] = attributes_dict[start][repo_name].get('commits_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if commit['committer']['login']:
		    head_name = commit['committer']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		if commit['author']['login']:
		    head_name = commit['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	events = db.events.find({"created_at": {"$lt": end,"$gte": start}})
	for event in events:
	    #print "debug 6"
	    if event:
		repo_name = event["repo"]["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['events_count'] = attributes_dict[start][repo_name].get('events_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if event['actor']['login']:
		    head_name = event['actor']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	issue_events = db.issue_events.find({"created_at": {"$lt": end,"$gte": start}})
	for issue_event in issue_events:
	    #print "debug 7"
	    if issue_event:
		if issue_event and issue_event['issue'] and issue_event['issue']['repository_url']:
		    repo_name = issue_event["issue"]["repository_url"].split('repos/mozilla/')[1].split('/')[0]
		    if attributes_dict[start].get(repo_name) == None:
			attributes_dict[start][repo_name] = {}
		    attributes_dict[start][repo_name]['issue_events_count'] = attributes_dict[start][repo_name].get('issue_events_count', 0) + 1
		    if attributes_dict[start][repo_name].get('user_list') == None:
			attributes_dict[start][repo_name]['user_list'] = []
		    if issue_event['actor'] and issue_event['actor']['login']:
			head_name = issue_event['actor']['login']
			if head_name not in attributes_dict[start][repo_name]['user_list']:
			    attributes_dict[start][repo_name]['user_list'].append(head_name)
			    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	releases = db.releases.find({"created_at": {"$lt": end,"$gte": start}})
	for release in releases:
	    #print "debug 8"
	    if release:
		repo_name = release["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['releases_count'] = attributes_dict[start][repo_name].get('releases_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if release and release.get('author') and release['author']['login']:
		    head_name = release['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	tags = db.tags.find({"created_at": {"$lt": end,"$gte": start}})
	for tag in tags:
	    #print "debug 9"
	    if tag:
		repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['tags_count'] = attributes_dict[start][repo_name].get('tags_count', 0) + 1

	branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
	for branch in branches:
	    #print "debug 10"
	    if branch:
		repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['branches_count'] = attributes_dict[start][repo_name].get('branches_count', 0) + 1

	start = start + delta_const
	end = end + delta_const

    #print "attributes_dict\n",attributes_dict
    #%%    
    num_time_slices = 12 #this will give num_time_slices+1
    num_time_slices = num_time_slices - 1 #this will give original num_time_slices (e.g. no of days from 1 to 3 is 3)
    start = start_const + num_time_slices*delta_const #start after num_time_slices of delta
    prev = start - num_time_slices*delta_const  #start before num_time_slices of delta
    end = start + delta_const
    varI = 0
    #print "nitk1",prev,start
    #print attributes_dict#.values()
    max_num_repos = 0
    while end <= final_end_const:
	
	prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date    
	  
	start_tmp = start #this is used in case there are time slices in which some repos are missing, so start_tmp will increment in those cases
	curr_max_num_repos = 0
	for item in prev_date_repos:#every entry in data will be for a particular repo. item is a repo
	  #print "varI",varI
	  
	  varI = varI + 1
	  tmp_data_list = [] #store N-1 time slices of data (attributes) for that repo
	  tmp_target_val = []
	  num_time_slices_obtained = 0 #for each item
	  while(prev<start_tmp):#while loop to get previous n time slices and to add it as features
	    prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date 
	    #print item
	    #print "prev, start_tmp ", prev, start_tmp
	    #print prev_date_repos
	    
	    dict_found_flag = 1
	    
	    while(item not in attributes_dict[prev]):
	      #print item
	      prev = prev + delta_const #skip that date
	      start_tmp = start_tmp + delta_const #increment upper limit as well,since a time slice was skipped
	      dict_found_flag = 0
	      if(prev in attributes_dict):
		if(item in attributes_dict[prev]):#if item is finally found in some date
		  dict_found_flag = 1
	      else:
		break
	    
	    if(dict_found_flag==1):  
	      tmp_data_list.extend([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
				  attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
				  attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
				  attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
				  attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
				  attributes_dict[prev][item].get('contributors_count', 0)])
	      num_time_slices_obtained = num_time_slices_obtained + 1
	      #print "xx",num_time_slices_obtained
			      
	    prev = prev + delta_const
	    if(prev>(final_end_const-delta_const)):#if no time slices and is about to go out of bounds
	      break
	    
	  #end of while(prev<start_tmp)
	  
	  #print item
	  #print prev,start
	  #print prev_date_repos
	  #at this point after exiting while loop, prev = start
	  #prev_date_repos =  attributes_dict[prev] #when prev = start, then don't add 'contributors_count' as feature as that is output to be predicted     
	  dict_found_flag = 1
	  if(prev<=(final_end_const-delta_const)):
	    while(item not in attributes_dict[prev]):
	      prev = prev + delta_const #skip that date
	      start_tmp = start_tmp + delta_const #increment upper limit as well,since a time slice was skipped
	      dict_found_flag = 0
	      if(prev in attributes_dict):
		if(item in attributes_dict[prev]):#if item is finally found in some date
		  dict_found_flag = 1
	      else:
		#print "debug 0.1"
		break
		    
	    if(dict_found_flag==1):   
	      tmp_target_val =attributes_dict[prev][item].get('contributors_count', 0)
	      #print "tmp_target_val",tmp_target_val
	      #tmp_data_list.extend([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
				  #attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
				  #attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
				  #attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
				  #attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0)])

	      #print "debug 0.2",attributes_dict[prev][item].get('contributors_count', 0)
	      num_time_slices_obtained = num_time_slices_obtained + 1
	      #print "debug 0.2",num_time_slices_obtained
	  #print "debug 0.0",prev
	  #print "num_time_slices_obtained",num_time_slices_obtained
	  if(num_time_slices_obtained == (num_time_slices+1)): #only if 6 (or n) time slices were obtained, then only add to ML data
	    #curr_prediction = attributes_dict[prev][item].get('contributors_count', 0)
	    #print item
	    #tmp_data_list.extend(item)
	    data_list.append(tmp_data_list)	
	    target_list.append(tmp_target_val)
	    list_of_all_Y_vals.extend([tmp_target_val])
	    
	    #data_target_list_dict[item] = {}
	    #data_target_list_dict[item][prev] = collections.defaultdict(list)
	    #data_target_list_dict[item][prev]['X'].append(tmp_data_list)
	    #data_target_list_dict[item][prev]['Y'].append(tmp_target_val)
	    #print data_list
	    #print "debug 1"
	    
	    curr_max_num_repos = curr_max_num_repos + 1
	    #print "debug 2"
	  prev = start - num_time_slices*delta_const  #start before num_time_slices of delta, reset prev for next iteration of for loop
	  start_tmp = start
	  #print "debug 3",prev
	  #print "debug 4",start
	#end of for item in prev_date_repos
	
	if(curr_max_num_repos>max_num_repos):
	  max_num_repos = curr_max_num_repos
	  
	start = start + delta_const
	prev = start - num_time_slices*delta_const  #start before num_time_slices of delta
	end = end + delta_const                
	
	#end of while end <= final_end_const
	
    #for key1 in data_target_list_dict:
      #for key2 in data_target_list_dict[key1]:     
	#list_of_all_Y_vals.extend(data_target_list_dict[key1][key2]['Y'])


    #change continuous values of target_list to discrete classes
    numOfRequiredClasses = 20
    
    if(numOfRequiredClasses>len(list_of_all_Y_vals)):
      numOfRequiredClasses = len(list_of_all_Y_vals)
    
    print target_list
    
    jenks_classes = jenks(target_list, numOfRequiredClasses)
    #print "jenks_classes",jenks_classes  
    #print data_target_list_dict
    
    list_of_all_Y_classes = classify(target_list, jenks_classes)
    print "list_of_all_Y_vals",list_of_all_Y_vals  
    print "list_of_all_Y_classes",list_of_all_Y_classes
    print "target_list",target_list
    target_list = list_of_all_Y_classes
    print "target_list",target_list    
    print "jenks_classes",jenks_classes  
    
    #print data_list 
    #print target_list
    #print data_target_list_dict
    #%%
    print "max num of repos processed in 'n'* delta time slices", max_num_repos
    print "\n\n"
    #print data_list 
    #print target_list
    #for repo_name in attributes_dict:
	#print repo_name
	#data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	#target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

    total_data_length = len(data_list)
    print total_data_length
    train_length = int(math.floor(0.8 * len(data_list))) #80% of data used for training
    data_train_list = np.array(data_list[0: train_length])
    target_train_list = np.array(target_list[0: train_length])
    data_test_list = np.array(data_list[train_length:])
    target_test_list = np.array(target_list[train_length:])
    
    
    #%%
    
    ## Run SVM classifier  
    print "data_test_list",data_test_list
    clf = svm.SVC(gamma=0.0001, C=200.)
    clf.fit(data_train_list, target_train_list)  
    target_predict_list = clf.predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "svm classifier_accuracy: ",classifier_accuracy
    
    ## Run random forest classifier  
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(data_train_list, target_train_list)  
    target_predict_list = clf.predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "random forest classifier_accuracy: ",classifier_accuracy
    
    ## Run gaussian naive bayes classifier
    clf = GaussianNB()
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "naive bayes classifier_accuracy: ",classifier_accuracy
    
    ## Run KNeighborsClassifier classifier
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf.fit(data_train_list, target_train_list)
    #target_predict_list = clf.predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "KNeighborsClassifier classifier_accuracy: ",classifier_accuracy

    ## Run SVC classifier
    clf = SVC(kernel="linear", C=0.025)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "SVC classifier_accuracy: ",classifier_accuracy

    ## Run GaussianProcessClassifier classifier
    #clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    #target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "GaussianProcessClassifier classifier_accuracy: ",classifier_accuracy

    ## Run DecisionTreeClassifier classifier
    clf = DecisionTreeClassifier(max_depth=5)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "DecisionTreeClassifier classifier_accuracy: ",classifier_accuracy

    ## Run RandomForestClassifier classifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=200, max_features=1)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "RandomForestClassifier classifier_accuracy: ",classifier_accuracy

    ## Run MLPClassifier classifier
    clf = MLPClassifier(alpha=1)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "MLPClassifier classifier_accuracy: ",classifier_accuracy

    ## Run AdaBoostClassifier classifier
    clf = AdaBoostClassifier()
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "AdaBoostClassifier classifier_accuracy: ",classifier_accuracy
    
    ## Run QuadraticDiscriminantAnalysis classifier
    #clf = QuadraticDiscriminantAnalysis()
    #target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "QuadraticDiscriminantAnalysis classifier_accuracy: ",classifier_accuracy
    
    #'''
    
  
  
  
  
  
  elif(resultSet == 4):
    ##Code for results set 4 # runs classification using prev time slice data and prev contri count and predicts current count
    #'''
    #%%
    
    print "Code for results set 4"
    ##start_const = datetime.datetime(2010, 06, 01, 0, 00, 00) #(2015, 06, 01, 0, 00, 00)
    ##final_end_const = datetime.datetime(2016, 7, 1, 0, 00, 00)#(2012, 7, 1, 0, 00, 00)#(2016, 7, 1, 0, 00, 00)
    ##delta_const = datetime.timedelta(days=45)
    start = start_const
    end = start + delta_const
    attributes_dict = {}
    data_list = []
    target_list = []
    varI = 0
    
    while end <= final_end_const:
	print start
	print end
	#print db
	print varI
	varI = varI + 1
	start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")#2016-05-03T20:26:43Z
	end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
	
	#items = db.pulls.find({"created_at": {"$lt": "2016-05-03T20:26:43Z","$gte": "2016-03-03T20:26:43Z"}})
	#print "debug 0.1"
	##print(db.collection_names())  
	attributes_dict[start] = {}
	#print attributes_dict
	items = db.pulls.find({"created_at": {"$lt": end ,"$gte": start}})#pull instances greater than start date and lesser than end date    
	for item in items:
	    #print "debug 0"
	    if item:
		repo_name = item['base']['repo']['name']
		#print item['created_at']
		#wait = input("PRESS ENTER TO CONTINUE.")
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['pulls_count'] = attributes_dict[start][repo_name].get('pulls_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['head'] and item['head']['user']:
		    head_name = item['head']['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		base_name = item['base']['user']['login']
		if head_name != base_name and base_name not in attributes_dict[start][repo_name]['user_list']:
		    attributes_dict[start][repo_name]['user_list'].append(base_name)
		    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    
	items = db.issue_comment.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 1"
	    if item:
		repo_name = item['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issue_comment_count'] = attributes_dict[start][repo_name].get('issue_comment_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	items = db.issues.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 2"
	    if item:
		repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['issues_count'] = attributes_dict[start][repo_name].get('issues_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	items = db.review_comments.find({"created_at": {"$lt": end,"$gte": start}})
	for item in items:
	    #print "debug 3"
	    if item:
		if item.get('repository_url'):
		    repo_name = item['repository_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['review_comments_count'] = attributes_dict[start][repo_name].get('review_comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if item and item['user'] and item['user']['login']:
		    head_name = item['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	comments = db.comments.find({"created_at": {"$lt": end,"$gte": start}})
	for comment in comments:
	    #print "debug 4"
	    if comment:
		repo_name = comment['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['comments_count'] = attributes_dict[start][repo_name].get('comments_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if comment['user']['login']:
		    head_name = comment['user']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	commits = db.commits.find({"created_at": {"$lt": end,"$gte": start}})
	for commit in commits:
	    #print "debug 5"
	    if commit:
		repo_name = commit['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['commits_count'] = attributes_dict[start][repo_name].get('commits_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if commit['committer']['login']:
		    head_name = commit['committer']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
		if commit['author']['login']:
		    head_name = commit['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	events = db.events.find({"created_at": {"$lt": end,"$gte": start}})
	for event in events:
	    #print "debug 6"
	    if event:
		repo_name = event["repo"]["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['events_count'] = attributes_dict[start][repo_name].get('events_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if event['actor']['login']:
		    head_name = event['actor']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	issue_events = db.issue_events.find({"created_at": {"$lt": end,"$gte": start}})
	for issue_event in issue_events:
	    #print "debug 7"
	    if issue_event:
		if issue_event and issue_event['issue'] and issue_event['issue']['repository_url']:
		    repo_name = issue_event["issue"]["repository_url"].split('repos/mozilla/')[1].split('/')[0]
		    if attributes_dict[start].get(repo_name) == None:
			attributes_dict[start][repo_name] = {}
		    attributes_dict[start][repo_name]['issue_events_count'] = attributes_dict[start][repo_name].get('issue_events_count', 0) + 1
		    if attributes_dict[start][repo_name].get('user_list') == None:
			attributes_dict[start][repo_name]['user_list'] = []
		    if issue_event['actor'] and issue_event['actor']['login']:
			head_name = issue_event['actor']['login']
			if head_name not in attributes_dict[start][repo_name]['user_list']:
			    attributes_dict[start][repo_name]['user_list'].append(head_name)
			    attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1

	releases = db.releases.find({"created_at": {"$lt": end,"$gte": start}})
	for release in releases:
	    #print "debug 8"
	    if release:
		repo_name = release["url"].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['releases_count'] = attributes_dict[start][repo_name].get('releases_count', 0) + 1
		if attributes_dict[start][repo_name].get('user_list') == None:
		    attributes_dict[start][repo_name]['user_list'] = []
		if release and release.get('author') and release['author']['login']:
		    head_name = release['author']['login']
		    if head_name not in attributes_dict[start][repo_name]['user_list']:
			attributes_dict[start][repo_name]['user_list'].append(head_name)
			attributes_dict[start][repo_name]['contributors_count'] = attributes_dict[start][repo_name].get('contributors_count', 0) + 1
	    

	tags = db.tags.find({"created_at": {"$lt": end,"$gte": start}})
	for tag in tags:
	    #print "debug 9"
	    if tag:
		repo_name = tag['zipball_url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name)==None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['tags_count'] = attributes_dict[start][repo_name].get('tags_count', 0) + 1

	branches = db.branches.find({"created_at": {"$lt": end,"$gte": start}})
	for branch in branches:
	    #print "debug 10"
	    if branch:
		repo_name = branch['commit']['url'].split('repos/mozilla/')[1].split('/')[0]
		if attributes_dict[start].get(repo_name) == None:
		    attributes_dict[start][repo_name] = {}
		attributes_dict[start][repo_name]['branches_count'] = attributes_dict[start][repo_name].get('branches_count', 0) + 1

	start = start + delta_const
	end = end + delta_const


    #%%    
    prev = start_const
    start = start_const + delta_const
    end = start + delta_const
    varI = 0
    #print attributes_dict#.values()
    while end <= final_end_const:
	#print 'nitk'
	##print prev
	##print start
	##print end
	#print db
	##print varI
	varI = varI + 1
	curr_date_repos =  attributes_dict[start] #dictionary of repos for a given date
	prev_date_repos =  attributes_dict[prev] #dictionary of repos for a given date
	#print curr_date_repos
	#print attributes_dict[start]
	for item in curr_date_repos: #iterate over keys, i.e. repos here
	  #print item
	  attributes_dict[start][item]['prev_contributors_count'] = attributes_dict[start][item].get('prev_contributors_count', 0)
	  #print "debug 1"
	  #print attributes_dict[start][item]
	  
	  if item in prev_date_repos:
	    attributes_dict[start][item]['prev_contributors_count'] = attributes_dict[prev][item]['contributors_count']
	    data_list.append([attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
			    attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
			    attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
			    attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
			    attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
			    attributes_dict[prev][item].get('contributors_count', 0)])
	    target_list.append(attributes_dict[start][item].get('contributors_count', 0))
	    
	    print "Data for presentation:;{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(prev, item,
	    attributes_dict[prev][item].get('pulls_count', 0), attributes_dict[prev][item].get('issue_comment_count', 0),
		      attributes_dict[prev][item].get('issues_count', 0), attributes_dict[prev][item].get('review_comments_count', 0),
		      attributes_dict[prev][item].get('comments_count', 0), attributes_dict[prev][item].get('commits_count', 0),
		      attributes_dict[prev][item].get('events_count', 0), attributes_dict[prev][item].get('issue_events_count', 0), 
		      attributes_dict[prev][item].get('releases_count', 0), attributes_dict[prev][item].get('tags_count', 0),
		      attributes_dict[prev][item].get('contributors_count', 0),attributes_dict[start][item].get('contributors_count', 0))


	prev = prev + delta_const
	start = start + delta_const
	end = end + delta_const    

    print data_list 
    print target_list
    #%%

    print data_list 
    print target_list
    #for repo_name in attributes_dict:
	#print repo_name
	#data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	#target_list.append(attributes_dict[repo_name].get('contributors_count', 0))  
    
    #change continuous values of target_list to discrete classes
    numOfRequiredClasses = 20
    list_of_all_Y_vals = target_list
    if(numOfRequiredClasses>len(list_of_all_Y_vals)):
      numOfRequiredClasses = len(list_of_all_Y_vals)
      
    jenks_classes = jenks(target_list, numOfRequiredClasses)
    #print "jenks_classes",jenks_classes  
    #print data_target_list_dict
    list_of_all_Y_classes = classify(target_list, jenks_classes)
    print "list_of_all_Y_vals",list_of_all_Y_vals  
    print "list_of_all_Y_classes",list_of_all_Y_classes
    print "target_list",target_list
    target_list = list_of_all_Y_classes
    print "target_list",target_list    
    print "jenks_classes",jenks_classes  
    
    #print data_list 
    #print target_list
    #print data_target_list_dict
    #%%
    #print "max num of repos processed in 'n'* delta time slices", max_num_repos
    print "\n\n"
    #print data_list 
    #print target_list
    #for repo_name in attributes_dict:
	#print repo_name
	#data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0), attributes_dict[repo_name].get('tags_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	##data_list.append([attributes_dict[repo_name].get('pulls_count', 0), attributes_dict[repo_name].get('issue_comment_count', 0), attributes_dict[repo_name].get('issues_count', 0), attributes_dict[repo_name].get('review_comments_count', 0), attributes_dict[repo_name].get('comments_count', 0), attributes_dict[repo_name].get('commits_count', 0), attributes_dict[repo_name].get('events_count', 0), attributes_dict[repo_name].get('issue_events_count', 0), attributes_dict[repo_name].get('releases_count', 0)])
	#target_list.append(attributes_dict[repo_name].get('contributors_count', 0))

    total_data_length = len(data_list)
    print total_data_length
    train_length = int(math.floor(0.8 * len(data_list))) #80% of data used for training
    data_train_list = np.array(data_list[0: train_length])
    target_train_list = np.array(target_list[0: train_length])
    data_test_list = np.array(data_list[train_length:])
    target_test_list = np.array(target_list[train_length:])
    
    
    #%%
    
    ## Run SVM classifier  
    print "data_test_list",data_test_list
    clf = svm.SVC(gamma=0.0001, C=200.)
    clf.fit(data_train_list, target_train_list)  
    target_predict_list = clf.predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "svm classifier_accuracy: ",classifier_accuracy
    
    ## Run random forest classifier  
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(data_train_list, target_train_list)  
    target_predict_list = clf.predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "random forest classifier_accuracy: ",classifier_accuracy
    
    ## Run gaussian naive bayes classifier
    clf = GaussianNB()
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "naive bayes classifier_accuracy: ",classifier_accuracy
    
    ## Run KNeighborsClassifier classifier
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf.fit(data_train_list, target_train_list)
    #target_predict_list = clf.predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "KNeighborsClassifier classifier_accuracy: ",classifier_accuracy

    ## Run SVC classifier
    clf = SVC(kernel="linear", C=0.025)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "SVC classifier_accuracy: ",classifier_accuracy

    ## Run GaussianProcessClassifier classifier
    #clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    #target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "GaussianProcessClassifier classifier_accuracy: ",classifier_accuracy

    ## Run DecisionTreeClassifier classifier
    clf = DecisionTreeClassifier(max_depth=5)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "DecisionTreeClassifier classifier_accuracy: ",classifier_accuracy

    ## Run RandomForestClassifier classifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=200, max_features=1)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "RandomForestClassifier classifier_accuracy: ",classifier_accuracy

    ## Run MLPClassifier classifier
    clf = MLPClassifier(alpha=1)
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "MLPClassifier classifier_accuracy: ",classifier_accuracy

    ## Run AdaBoostClassifier classifier
    clf = AdaBoostClassifier()
    target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    print "target_test_list",target_test_list
    print "target_predict_list",target_predict_list

    classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    print "AdaBoostClassifier classifier_accuracy: ",classifier_accuracy
    
    ## Run QuadraticDiscriminantAnalysis classifier
    #clf = QuadraticDiscriminantAnalysis()
    #target_predict_list = clf.fit(data_train_list, target_train_list).predict(data_test_list)
    #print "target_test_list",target_test_list
    #print "target_predict_list",target_predict_list

    #classifier_accuracy = accuracy_score(target_test_list, target_predict_list)
    #print "QuadraticDiscriminantAnalysis classifier_accuracy: ",classifier_accuracy
    
    #'''

  

if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
