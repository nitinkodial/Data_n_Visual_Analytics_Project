mongoimport --db github_database --collection review_comments --drop --file review_comments.json --jsonArray
mongoimport --db github_database --collection assignees --drop --file assignees.json --jsonArray
mongoimport --db github_database --collection branches --drop --file branches.json --jsonArray
mongoimport --db github_database --collection comments --drop --file comments.json --jsonArray
mongoimport --db github_database --collection contents --drop --file contents.json --jsonArray
mongoimport --db github_database --collection contributors --drop --file contributors.json --jsonArray
mongoimport --db github_database --collection deployments --drop --file deployments.json --jsonArray
mongoimport --db github_database --collection downloads --drop --file downloads.json --jsonArray
mongoimport --db github_database --collection events --drop --file events.json --jsonArray
mongoimport --db github_database --collection forks --drop --file forks.json --jsonArray
mongoimport --db github_database --collection git_refs --drop --file git_refs.json --jsonArray
mongoimport --db github_database --collection issue_comment --drop --file issue_comment.json --jsonArray
mongoimport --db github_database --collection issues --drop --file issues.json --jsonArray
mongoimport --db github_database --collection labels --drop --file labels.json --jsonArray
mongoimport --db github_database --collection languages --drop --file languages.json --jsonArray
mongoimport --db github_database --collection milestones --drop --file milestones.json --jsonArray
mongoimport --db github_database --collection organization --drop --file organization.json --jsonArray
mongoimport --db github_database --collection public_members --drop --file public_members.json --jsonArray
mongoimport --db github_database --collection pulls --drop --file pulls.json --jsonArray
mongoimport --db github_database --collection releases --drop --file releases.json --jsonArray
mongoimport --db github_database --collection repositories --drop --file repositories.json --jsonArray
mongoimport --db github_database --collection stargazers --drop --file stargazers.json --jsonArray
mongoimport --db github_database --collection subscribers --drop --file subscribers.json --jsonArray
mongoimport --db github_database --collection tags --drop --file tags.json --jsonArray
mongoimport --db github_database --collection issue_events --drop --file issue_events.json --jsonArray
mongoimport --db github_database --collection commits --drop --file commits.json --jsonArray


# // use dva_github_data  //db name