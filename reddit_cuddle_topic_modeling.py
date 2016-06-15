import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
import praw
from nltk.stem.porter import *
import re


user_agent = ("[your user agent]")    # change this to your user agent
r = praw.Reddit(user_agent = user_agent)
# USask_Cuddle_Buddies will serve as outliers
# subreddit_lst = ["cuddle_with_me", "Cuddles", "cuddlebuddies", "USask_Cuddle_Buddies"]
subreddit_lst = ["cuddlebuddies"]
reddit_prefix = "https://www.reddit.com/r/"

limit_num = 5
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

posts = {}

for st in subreddit_lst:
  subreddit = r.get_subreddit(st)
  url_prefix = reddit_prefix+st+"/"
  for s in subreddit.get_hot(limit = limit_num):
    sid = s.id
    posts[sid] = {}
    posts[sid]["title"] = s.title
    posts[sid]["text"] = s.title+' '+s.selftext
    posts[sid]["score"] = s.score
    posts[sid]["comments_num"] = s.num_comments
    posts[sid]["url"] = url_prefix+sid
print len(posts.keys())
