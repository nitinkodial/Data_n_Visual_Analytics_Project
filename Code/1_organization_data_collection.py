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

# LOG_FILENAME = 'general.log'
# FORMAT = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
# logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)

ORG_NAMES = ["mozilla"]


def getDataFromUrl(url):
    try:
        if url:
            url = url.split('{')[0] + "?access_token=" + ACCESS_TOKEN + '&per_page=100'
        logging.debug("url = " + url)
        response = requests.get(url)
        dataJson = response.json()
        link = response.headers.get('link', None)
        i = 2
        while link and 'rel="next"' in link:
            response = requests.get(url + '&page={0}'.format(str(i)))
            link = response.headers.get('link', None)
            dataJson.extend(response.json())
            i += 1
        # logging.debug("status_code = " + str(response.status_code) + " : dataJson = " + json.dumps(dataJson))
        if response.status_code == 200 and 'message' not in dataJson:
            return dataJson
        elif response.status_code >= 500 or \
                                'X-RateLimit-Remaining' in response.headers and \
                                int(response.headers["X-RateLimit-Remaining"]) == 0:
            # logging.info("Rate Limit Hit : status_code = " +
            #              str(response.status_code) +
            #              " : dataJson = " +
            #              json.dumps(dataJson) +
            #              " : X-RateLimit-Remaining = " +
            #              response.headers["X-RateLimit-Remaining"] +
            #              " : X-RateLimit-Reset = " + response.headers["X-RateLimit-Reset"])
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
            # logging.info("status_code = " + str(response.status_code) + " : dataJson = " + json.dumps(dataJson))
            print ("status_code = " + str(response.status_code) + " : dataJson = " + json.dumps(dataJson))
            return {}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as inst:
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst
        print "other error"
        # logging.error("Error in API Handling")
        # logging.error(traceback.format_exc())
        # raise Exception("Network Error", "retry")


def getOrgDataRepo(organization):
    try:
        org_url = API_URL + organization
        org_data = getDataFromUrl(org_url)
        if org_data == {}:
            # logging.error("Error in getting organization data")
            # logging.error(traceback.format_exc())
            raise Exception("Data Get Error", "retry")
        repos_data = {}
        public_members_data = {}
        if "repos_url" in org_data:
            repos_url = org_data["repos_url"]
            try:
                repos_data = getDataFromUrl(repos_url)
                for data in repos_data:
                    url = [prop for prop in data if prop != 'url' and '_url' in prop]
                    for prop in url:
                        if prop in data:
                            induvidual_url = data[prop]
                            try:
                                induvidual_property_data = getDataFromUrl(induvidual_url)
                                patch_data,review_comments_data,diff_data = None, None, None
                                if 'pulls' in prop and induvidual_property_data:
                                    print prop, induvidual_property_data
                                    i = 0
                                    for prop_prop in induvidual_property_data:
                                        print prop_prop
                                        if 'review_comments_url' in prop_prop:
                                            review_comments_url = prop_prop['review_comments_url']
                                            print review_comments_url
                                            review_comments_data = getDataFromUrl(review_comments_url)
                                            try:
                                                if review_comments_data:
                                                    json_file_name = "review_comments" + ".json"
                                                    print json_file_name
                                                    with open(json_file_name, "a+") as f:
                                                        review_comments_data = json.dumps(review_comments_data)
                                                        print review_comments_data
                                                        f.write(review_comments_data)
                                                        f.close()
                                            except (KeyboardInterrupt, SystemExit):
                                                raise
                                        if 'patch_url' in prop_prop:
                                            patch_url = prop_prop['patch_url']
                                            patch_data = urllib.urlopen(patch_url).read()
                                            try:
                                                if patch_data:
                                                    json_file_name = '_'.join(patch_url.split('github.com/')[1].split('/'))
                                                    i += 1
                                                    print json_file_name
                                                    with open(json_file_name, "a+") as f:
                                                        print patch_data
                                                        f.write(patch_data)
                                                        f.close()
                                            except (KeyboardInterrupt, SystemExit):
                                                raise
                                        if 'diff_url' in prop_prop:
                                            diff_url = prop_prop['diff_url']
                                            diff_data = urllib.urlopen(diff_url).read()
                                            try:
                                                if diff_data:
                                                    json_file_name = ('_').join(diff_url.split('github.com/')[1].split('/'))
                                                    i += 1
                                                    print json_file_name
                                                    with open(json_file_name, "a+") as f:
                                                        print diff_data
                                                        f.write(diff_data)
                                                        f.close()
                                            except (KeyboardInterrupt, SystemExit):
                                                raise
                                if 'language' in prop or 'subscribers' in prop or 'assignees' in prop or 'contributors' in prop or 'stargazers' in prop:
                                    induvidual_property_data = [{"repo_id": data['id'], "value": induvidual_property_data}]

                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except Exception as inst:
                                    # logging.error("Error in getting " + prop + " data")
                                    # logging.error(traceback.format_exc())
                                print inst.args
                                print type(inst)
                                print inst
                                print ("Error in getting " + prop + " data")
                                raise Exception("Data Get Error", "retry")
                            if induvidual_property_data:
                                try:
                                    json_file_name = prop.split("_url")[0] + ".json"
                                    is_new_file = (os.stat(json_file_name).st_size == 0)
                                    with open(json_file_name, "a+") as f:
                                        induvidual_property_data = json.dumps(induvidual_property_data)
                                        if is_new_file:
                                            f.write(induvidual_property_data[:-1]+',')
                                        elif repos_index == len(repos_data) - 1:
                                            f.write(induvidual_property_data[1:-1]+',')
                                        else:
                                            f.write(json.dumps(induvidual_property_data)[1:])
                                        f.close()
                                except (KeyboardInterrupt, SystemExit):
                                    raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                # logging.error("Error in getting repository data")
                # logging.error(traceback.format_exc())
                print "Error in getting repository data"
                raise Exception("Data Get Error", "retry")
        if "public_members_url" in org_data:
            public_members_url = org_data["public_members_url"]
            try:
                public_members_data = getDataFromUrl(public_members_url)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                # logging.error("Error in getting public members data")
                # logging.error(traceback.format_exc())
                print "Error in getting public members data"
                raise Exception("Data Get Error", "retry")

    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        # logging.error("Error in getting data")
        # logging.error(traceback.format_exc())
        raise Exception("Data Get Error", "retry")
    try:
        with open("organization.json", "a+") as f:
            org_data = json.dumps(org_data)
            f.write(org_data)
            f.close()
    except (KeyboardInterrupt, SystemExit):
        raise

    try:
        with open("repositories.json", "a+") as f:
            repos_data = json.dumps(repos_data)
            f.write(repos_data)
            f.close()
    except (KeyboardInterrupt, SystemExit):
        raise

    try:
        with open("public_members.json", "a+") as f:
            public_members_data = json.dumps(public_members_data)
            f.write(public_members_data)
            f.close()
    except (KeyboardInterrupt, SystemExit):
        raise


def main():
    # logging.info("System startup")
    list_of_organizations = set(ORG_NAMES)
    for organization in list_of_organizations:
        try:
            getOrgDataRepo(organization)
        except Exception as e:
            if e.args[0] == "Data Get Error":
                with open(FAILED_ORGANIZATION_GET_LOG, "a+") as log:
                    log.write(organization + "\n")
        except KeyboardInterrupt as e:
            with open(FAILED_ORGANIZATION_GET_LOG, "a+") as log:
                log.write(organization + "\n")
            logging.info("Exited by user using KeyboardInterrupt")
            sys.exit(0)
        except SystemExit as e:
            with open(FAILED_ORGANIZATION_GET_LOG, "a+") as log:
                log.write(organization + "\n")
            logging.info("Exited by user using SystemExit")
            sys.exit(0)
        except:
            logging.error("Unexpected error occurred")
            logging.error(traceback.format_exc())
            with open(FAILED_ORGANIZATION_GET_LOG, "a+") as log:
                log.write(organization + "\n")
            sys.exit(0)
        else:
            with open(SUCCESS_ORGANIZATION_LOG, "a+") as log:
                log.write(organization + "\n")


if __name__ == "__main__":
    main()





