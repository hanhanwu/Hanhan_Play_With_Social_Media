'''
I am using Spark cloud, some features here can only be used in the Python Notebook in Spark
'''

# cell 1
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


# cell 2
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')


# cell 3
import praw
import re

user_agent = ("[use your own user agent]")
r = praw.Reddit(user_agent = user_agent)
# USask_Cuddle_Buddies will serve as outliers
# subreddit_lst = ["cuddle_with_me", "Cuddles", "cuddlebuddies", "USask_Cuddle_Buddies"]

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


# cell 4
from nltk.stem.porter import *
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

def clean_review(review_line):
  replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
  review_text = unicodedata.normalize("NFKD", review_line).encode('ascii','ignore').translate(replace_punctuation).split()
  review_words = [w.lower() for w in review_text if w not in stopwords]

  return review_words

all_words1 = title_text_rdd.map(clean_review)  # title_text, each element in RDD is a post
all_words2 = text_only_rdd.map(clean_review)   # text_noly, each element in RDD is a post


# cell 5
# convert to bag of words with tf-idf score, then normalize the scores into [0,1] range
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.feature import Normalizer

def get_tfidf_features(txt_rdd):
    hashingTF = HashingTF()
    tf = hashingTF.transform(txt_rdd)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    return tfidf
  
nor = Normalizer(1)
  
words_bag1 = get_tfidf_features(all_words1)
nor_words_bag1 = nor.transform(words_bag1)

words_bag2 = get_tfidf_features(all_words2)
nor_words_bag2 = nor.transform(words_bag2)


# cell 6
# LDA Modeling
## REFERENCE: http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

corpus = nor_words_bag1.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
ldaModel = LDA.train(corpus, k=5)

print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(5):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))
