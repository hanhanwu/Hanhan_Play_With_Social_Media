# This is the most convenient and easy-to-use toppic modeling method I have ever tried!! It simply returns top topics

# cell 1 - collect reddit posts
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


# cell 2 - extract features for NMF and LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

n_top_words = 20
n_topics = 10
n_features = 1000
n_samples = len(posts)

def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print "Topic #%d:" % topic_idx
    print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print 
    

print "Extracting tf-idf features for NMF..."
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(posts)

print "Extracting tf features for LDA..."
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(posts)


# cell 3 - Using NMF to get top topics
print "Fitting the NMF model with tf-idf features," "n_samples=%d and n_features=%d..." % (n_samples, n_features)
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

print "\nTopics in NMF model:" 
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)



# cell 4 - Using LDA to get top topics
print "Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features)
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print "\nTopics in LDA model:"
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
