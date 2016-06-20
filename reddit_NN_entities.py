# cell 1
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')


# cell 2
import praw
import re

user_agent = ("cuddle_analysis 1.01")
r = praw.Reddit(user_agent = user_agent)

subreddit_lst = ["cuddlebuddies"]
reddit_prefix = "https://www.reddit.com/r/"

limit_num = 1000

posts = []

for st in subreddit_lst:
  subreddit = r.get_subreddit(st)
  url_prefix = reddit_prefix+st+"/"
  # use hot topics first
  for s in subreddit.get_hot(limit = limit_num):
    posts.append(s.title + " " + s.selftext)
    
print len(posts)


# cell 3
# Get NN entities

def get_NN_entities(post):
    sentences = nltk.tokenize.sent_tokenize(post)
    token_sets = [nltk.tokenize.word_tokenize(s) for s in sentences]
    pos_tagged_token_sets = [nltk.pos_tag(t) for t in token_sets]
    pos_tagged_tokens = [t for v in pos_tagged_token_sets for t in v]
    
    all_entities = []
    previous_pos = None
    current_entities = []
    for (entity, pos) in pos_tagged_tokens:
        if previous_pos == pos and pos.startswith('NN'):
            current_entities.append(entity.lower())
        elif pos.startswith('NN'):
            if current_entities != []:
                all_entities.append(' '.join(current_entities))
            current_entities = [entity.lower()]
        previous_pos = pos
    return all_entities

  
def myf(x):
  return x


posts_rdd = sc.parallelize(posts)
reddit_entities = posts_rdd.map(get_NN_entities).flatMap(myf).collect()


# cell 4 - Get top 50 NN entities
import operator

entity_dct = {}

for ety in reddit_entities:
  entity_dct[ety] = entity_dct.get(ety, 0) + 1
  
sorted_entities = sorted(entity_dct.items(), key=operator.itemgetter(1), reverse=True)

ct = 50
for i in range(ct):
  print sorted_entities[i]
