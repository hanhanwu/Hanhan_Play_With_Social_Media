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



# cell 3 - get NN interactions, NNVBNN interactions
def get_NN_interactions(txt):
    sentences = nltk.tokenize.sent_tokenize(txt)
    token_sets = [nltk.tokenize.word_tokenize(s) for s in sentences]
    pos_tagged_tokens = [nltk.pos_tag(t) for t in token_sets]
    
    all_interactions = []
    for sentence in pos_tagged_tokens:
        all_entities = []
        previous_pos = None
        current_entities = []
        for (entity, pos) in sentence:
            if previous_pos == pos and pos.startswith('NN'):
                current_entities.append(entity)
            elif pos.startswith('NN'):
                if current_entities != []:
                    all_entities.append((' '.join(current_entities), pos))
                current_entities = [entity]
            previous_pos = pos
        
        if len(all_entities) >= 1:
            all_interactions.append(all_entities)
        else:
            all_interactions.append([])
            
    assert len(all_interactions) == len(sentences)
    
    return dict(all_interactions = all_interactions,
                sentences = sentences)
  
  
def get_NNVBNN_interactions(txt):
    sentences = nltk.tokenize.sent_tokenize(txt)
    token_sets = [nltk.tokenize.word_tokenize(s) for s in sentences]
    pos_tagged_tokens = [nltk.pos_tag(t) for t in token_sets]
    
    all_interactions = []
    for sentence in pos_tagged_tokens:
        all_entities = []        
        current_entities = []
        
        i = 2
        while i < len(sentence):
            fst_previous_pos = sentence[i-2][1]
            snd_previous_pos = sentence[i-1][1]
            current_pos = sentence[i][1]
            if fst_previous_pos.startswith('NN') and snd_previous_pos.startswith('VB') and current_pos.startswith('NN'):
                current_entities.append(sentence[i-2][0])
                current_entities.append(sentence[i-1][0])
                current_entities.append(sentence[i][0])
                all_entities.append((' '.join(current_entities), 'NNVBNN'))
                current_entities = []
            i += 1
             
        
        if len(all_entities) >= 1:
            all_interactions.append(all_entities)
        else:
            all_interactions.append([])
            
    assert len(all_interactions) == len(sentences)
    
    return dict(all_interactions = all_interactions,
                sentences = sentences)
  
  
reddit_NN_interactions = posts_rdd.map(get_NN_interactions).collect()
reddit_NNVBNN_interactions = posts_rdd.map(get_NNVBNN_interactions).collect()


# cell 4
reddit_NNVBNN_interactions[1]
