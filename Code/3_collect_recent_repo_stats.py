#!/usr/bin/python
import json
import requests
import time
import logging
import traceback
import urllib
import sqlite3
import os
import sys

FAILED_ORGANIZATION_GET_LOG = "failed-organization-get.log"
SUCCESS_ORGANIZATION_LOG = "success-organization.log"
API_URL = "https://api.github.com/orgs/"
ACCESS_TOKEN = "880991ffcdb663596fdb4f10c8c295a2f3d08711"

ORG_NAMES = ["mozilla"]
repo_list = []
property_list = ["contributors","commit_activity","code_frequency","participation", "punch_card"]


def getDataFromUrl(url):
    try:
        if url:
            url = url.split('{')[0] + "?access_token=" + ACCESS_TOKEN + '&per_page=100'
        logging.debug("url = " + url)
        print url
        response = requests.get(url)
        dataJson = response.json()
        link = response.headers.get('link', None)
        i = 2
        while link and 'rel="next"' in link:
            response = requests.get(url + '&page={0}'.format(str(i)))
            link = response.headers.get('link', None)
            dataJson.extend(response.json())
            i += 1
        
        if response.status_code == 200 and 'message' not in dataJson:
            return dataJson
        elif response.status_code >= 500 or \
                                'X-RateLimit-Remaining' in response.headers and \
                                int(response.headers["X-RateLimit-Remaining"]) == 0:
            print ("Rate Limit Hit : status_code = " +
                         str(response.status_code) +
                         " : dataJson = " +
                         json.dumps(dataJson) +
                         " : X-RateLimit-Remaining = " +
                         response.headers["X-RateLimit-Remaining"] +
                         " : X-RateLimit-Reset = " + response.headers["X-RateLimit-Reset"])
            time.sleep(int(response.headers["X-RateLimit-Reset"]) - int(time.time()) + 10)
            return getDataFromUrl(url)
        else:
            if response.status_code == 202:
                print '202', 'sleeping'
                time.sleep(60*10)
            print ("status_code = " + str(response.status_code) + " : dataJson = " + json.dumps(dataJson))
            return getDataFromUrl(url)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as inst:
        print type(inst)     
        print inst.args      
        print inst
        print "other error"



def main():
    list_of_organizations = set(ORG_NAMES)
    collect_url = "https://api.github.com/repos/mozilla/"
    repos_data_url = "https://api.github.com/orgs/mozilla/repos"
    repos_data = getDataFromUrl(repos_data_url)
    for repo in repos_data:
            print repo["name"]
            repo_list.append(repo["name"])
    for repo in repo_list:
        for prop in property_list:
            stat_url = collect_url + repo + "/stats/" + prop
            stat_data = json.dumps(getDataFromUrl(stat_url))
            time.sleep(60*10)           
            try:
                stat_data = json.dumps(getDataFromUrl(stat_url))
                json_file_name = prop + ".json"
                with open(json_file_name, "a+") as f:
                    f.write(stat_data)
                    f.close()
            except (KeyboardInterrupt, SystemExit):
                raise
                    
if __name__ == "__main__":
    main()





