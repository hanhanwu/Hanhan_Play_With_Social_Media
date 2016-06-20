# I'm using Spark Python Notebook, some features can only be found in this Spark Cloud

# cell 1
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')


# cell 2
import praw
import re

user_agent = ("[use your own user agent]")
r = praw.Reddit(user_agent = user_agent)

subreddit_lst = ["cuddlebuddies"]
reddit_prefix = "https://www.reddit.com/r/"

limit_num = 1000

posts = {}

for st in subreddit_lst:
  subreddit = r.get_subreddit(st)
  url_prefix = reddit_prefix+st+"/"
  # use hot topics first
  for s in subreddit.get_hot(limit = limit_num):
    sid = s.id
    posts[sid] = {}
    posts[sid]["title"] = s.title
    posts[sid]["text"] = s.selftext
    
print len(posts.keys())


# cell 3
import string
import unicodedata

stopwords = nltk.corpus.stopwords.words('english')
lst_has_title = []
lst_has_no_title = []

for v in posts.values():
  lst_has_title.append(v['title'] + ' ' + v['text'])
  lst_has_no_title.append(v['text'])

title_text_rdd = sc.parallelize(lst_has_title)
text_only_rdd = sc.parallelize(lst_has_no_title)

def clean_post(post_line):
  replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
  post_text = unicodedata.normalize("NFKD", post_line).encode('ascii','ignore').translate(replace_punctuation).split()
  post_words = [w.lower() for w in post_text if w.lower() not in stopwords]

  return post_words

cleaned_post = title_text_rdd.map(clean_post)  # title & text, each element in RDD is a post


# cell 4
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.clustering import KMeans
import numpy as np

def generate_word2vec_model(doc):
    return Word2Vec().setVectorSize(10).setSeed(410).fit(doc)
  
def generate_kmeans_model(rdd, k):
    return KMeans.train(rdd, k, maxIterations=10,
                                initializationMode="random", seed=410, initializationSteps=5, epsilon=1e-4)
  
word2vec_model = generate_word2vec_model(cleaned_post)
mv = word2vec_model.getVectors()  # this is a dictionary, the key is a word, the value is a list of number represent this word

words_array = np.array(mv.values())
k = 5
words_rdd = sc.parallelize(words_array)
kmeans_model = generate_kmeans_model(words_rdd, k)

unique_words = mv.keys()
kmeans_predicts = []
for unique_word in unique_words:
  vec = word2vec_model.transform(unique_word)
  kmeans_predict = kmeans_model.predict(vec)
  kmeans_predicts.append((unique_word, kmeans_predict))
  
  
# cell 5
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for itm in kmeans_predicts:
  if itm[1] == 1: c1.append(itm[0])
  elif itm[1] == 2: c2.append(itm[0])
  elif itm[1] == 3: c3.append(itm[0])
  elif itm[1] == 4: c4.append(itm[0])
  elif itm[1] == 5: c5.append(itm[0])
  
  
# cell 6
print c1
print c2
print c3
print c4
print c5



# try stemming when cleaning the words
# cell 7
# Do words stemming, by unify same words that have different format, see whether it the clusters make more sense
from nltk.stem.porter import *
import string
import unicodedata

stopwords = nltk.corpus.stopwords.words('english')
stemmer = PorterStemmer()
lst_has_title = []
lst_has_no_title = []

for v in posts.values():
  lst_has_title.append(v['title'] + ' ' + v['text'])
  lst_has_no_title.append(v['text'])

title_text_rdd = sc.parallelize(lst_has_title)
text_only_rdd = sc.parallelize(lst_has_no_title)

def clean_post(post_line):
  replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
  post_text = unicodedata.normalize("NFKD", post_line).encode('ascii','ignore').translate(replace_punctuation).split()
  post_words = [w.lower() for w in post_text if w.lower() not in stopwords]

  return [stemmer.stem(w) for w in post_words]

cleaned_post = title_text_rdd.map(clean_post)  # title & text, each element in RDD is a post



# cell 8
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.clustering import KMeans
import numpy as np

def generate_word2vec_model(doc):
    return Word2Vec().setVectorSize(10).setSeed(410).fit(doc)
  
def generate_kmeans_model(rdd, k):
    return KMeans.train(rdd, k, maxIterations=10,
                                initializationMode="random", seed=410, initializationSteps=5, epsilon=1e-4)
  
word2vec_model = generate_word2vec_model(cleaned_post)
mv = word2vec_model.getVectors()  # this is a dictionary, the key is a word, the value is a list of number represent this word

words_array = np.array(mv.values())
k = 5
words_rdd = sc.parallelize(words_array)
kmeans_model = generate_kmeans_model(words_rdd, k)

unique_words = mv.keys()
kmeans_predicts = []
for unique_word in unique_words:
  vec = word2vec_model.transform(unique_word)
  kmeans_predict = kmeans_model.predict(vec)
  kmeans_predicts.append((unique_word, kmeans_predict))
  

# cell 9
# Get each word post index and cluster index
def myf(x):
  return x

indexed_text = cleaned_post.zipWithIndex().map(lambda (words, idx): (idx, words))
indexed_words = indexed_text.flatMapValues(myf).map(lambda (idx, w): (w, idx))  # word and post index

cluster_lookup = sc.parallelize(kmeans_predicts) # word and cluster index

word_pidx_cidx = indexed_words.join(cluster_lookup)  # word, post index, cluster index



# cell 10
# Get post histogram, showing each post words distribution in the clusters
from pyspark.mllib.linalg import SparseVector

# each item in this list shows clusters of words in each post
pidx_cidx = word_pidx_cidx.map(lambda (w, (pidx, cidx)): (pidx, cidx)).groupByKey().mapValues(list).coalesce(1)

def get_histogram(t):
  pidx = t[0]
  cidx_lst = t[1]
  unique_cidxs = set(cidx_lst)
  unique_cidxs = list(unique_cidxs)
  unique_cidxs.sort()
  total_clusters = 5
  cluster_records = np.zeros(total_clusters)
  for cidx in cidx_lst:
      cluster_records[cidx] += 1
  sum_records = np.sum(cluster_records)
  l1_cluster_records = [x/sum_records for x in cluster_records]
  sparse_records = [x for x in l1_cluster_records if x > 0]
  sp_size = total_clusters
  sp = SparseVector(sp_size, unique_cidxs, sparse_records)
  return pidx, sp

posts_histogram = pidx_cidx.map(get_histogram)
posts_histogram.collect()[0]
