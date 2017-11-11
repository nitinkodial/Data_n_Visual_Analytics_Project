# Data_n_Visual_Analytics_Project_OssAssist
OssAssist- A System to Assist Open Source Organizations Plan and Forecast Projects: Achieved a R-squared score of 74% and a mean square error value of 17 in predicting the number of contributors to open source projects using machine learning techniques and visualized key project performance indicators.

All the software deliverables are inside the Code folder

The project can be run in two parts:

Preliminary steps:

The script 1_organization_data_collection.py uses github api, collects data and makes about 22 .json files for repository attributes
such as comments.json, issues.json. The total data file size is about 6GB. Hence a sample of the .json collected is provided in the sample_data folder

2_clean_json_file.py - cleans data to be fed into mongodb

3_collect_recent_repo_stats.py - collects recent statistics file related to each repository (Contains activity traces for a repository for about a month.
Running this script will create .json files such as users.json, commit_activity.json, participation.json) 

4_feed_json_to_mongo.sh - Once data collection is done, simply execute this .sh file to feed the json into MongoDB.


ML Part

5_ml_classification_regression.py - Runs Lasso, Elastic net regression over single time slice, n-1 time slices
Also divides contributors into classes and runs classification algorithms such as Naive Bayes and Random Forest for the data read from the repository.


Visualization part

In the folder templatevamp-flask, run the script main.py - the flask server script that renders the HTML pages

The main.py renders the statistics, interactive horizontal bar chart, HTML table containing prediction results immediately.

However, for the interactive treemap, it runs a script that takes the launch date(March 2016) as input and calculates the values needed for treemap

Since it crunches data for over a year real time, rendering treemap takes time.

