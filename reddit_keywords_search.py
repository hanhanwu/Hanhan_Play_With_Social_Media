# Search Method 1
import praw
import re

user_agent = ("cuddle_analysis 1.01")
r = praw.Reddit(user_agent = user_agent)

subreddit_lst = ["cuddlebuddies"]
reddit_prefix = "https://www.reddit.com/r/"

limit_num = 1000

posts = {}

for st in subreddit_lst:
  subreddit = r.get_subreddit(st)
  url_prefix = reddit_prefix+st+"/"
  # I tried to set period and author here, the search results were not accurate at all..
  x = r.search('seattle', subreddit)
  for e in x:
    print e
    print url_prefix+e.id
    
    

# Search Method 2 - more flexibility, cano control where to search
import praw
import re

user_agent = ("cuddle_analysis 1.01")
r = praw.Reddit(user_agent = user_agent)

subreddit_lst = ["cuddlebuddies"]
reddit_prefix = "https://www.reddit.com/r/"

limit_num = 1000

posts = {}

for st in subreddit_lst:
  subreddit = r.get_subreddit(st)
  url_prefix = reddit_prefix+st+"/"
  for s in subreddit.get_new(limit = limit_num):
    sid = s.id
    posts[sid] = {}
    posts[sid]["title"] = s.title.lower()
    posts[sid]["text"] = s.title.lower()+' '+s.selftext.lower()
    posts[sid]["score"] = s.score
    posts[sid]["comments_num"] = s.num_comments
    posts[sid]["url"] = url_prefix+sid
print len(posts.keys())

key_terms = ['seattle']

for v in posts.values():
  for key_term in key_terms:
    if key_term in v['title']:
      print v['title']
      print v['url']
      print
