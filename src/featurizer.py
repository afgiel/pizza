from collections import Counter
import math
import numpy as np

import utils

class Featurizer():

  def __init__(self, params):
    self.params = params

  def select_features(self, data):
    self.feature_map = SELECT_FUNCS[self.params.selector](self, data) 

  def featurize(self, data):
    return FEAT_FUNCS[self.params.featurizer](self, data)

################
# SELECT FUNCS #
################

  def count(self, data):
    print 'COUNTING POSTS'
    body_word_count = {}
    title_word_count = {}
    body_doc_count = {}
    title_doc_count = {}
    tokenized_bodies = []
    tokenized_titles = []
    labels = utils.get_labels_from_post_list(data)
    all_body_tokens = set()
    all_title_tokens = set()
    for i in range(len(data)):
      post = data[i]
      label = labels[i]
      body_tokens, title_tokens = utils.get_post_tokens(post)
      tokenized_bodies.append(body_tokens)
      tokenized_titles.append(title_tokens)
      for token in body_tokens:
        if token not in body_word_count:
          body_word_count[token] = Counter()
        body_word_count[token][label] += 1
        if label not in body_doc_count: 
          body_doc_count[label] = Counter()
        body_doc_count[label][token] += 1 
        all_body_tokens.add(token)
      for token in title_tokens:
        if token not in title_word_count:
          title_word_count[token] = Counter()
        title_word_count[token][label] += 1
        if label not in title_doc_count: 
          title_doc_count[label] = Counter()
        title_doc_count[label][token] += 1 
        all_title_tokens.add(token)
    self.body_word_count = body_word_count
    self.title_word_count = title_word_count
    self.body_doc_count = body_doc_count
    self.title_doc_count = title_doc_count
    self.all_body_tokens = all_body_tokens
    self.all_title_tokens = all_title_tokens
    return tokenized_bodies, tokenized_titles, labels


  def select_top_mi_ngrams(self, data):
    # count 
    tokenized_bodies, tokenized_titles, labels = self.count(data)
    body_mi = Counter() 
    title_mi = Counter()
    num_body_tokens = sum([sum(self.body_word_count[x].values()) for x in self.body_word_count])
    for token in self.all_body_tokens:
      body_mi[token] = self.compute_mi(token, self.body_word_count, self.body_doc_count, num_body_tokens, labels)
    num_title_tokens = sum([sum(self.title_word_count[x].values()) for x in self.title_word_count]) 
    for token in self.all_title_tokens:
      title_mi[token] = self.compute_mi(token, self.title_word_count, self.title_doc_count, num_title_tokens, labels)  
    top_body_tokens = body_mi.most_common(self.params.num_body_tokens)
    top_title_tokens = title_mi.most_common(self.params.num_title_tokens)
    print top_body_tokens[:10]
    print top_title_tokens[:10]
    self.ngram_body_features = {}
    self.ngram_title_features = {}
    index = 1
    for token, count in top_body_tokens:
      self.ngram_body_features[token] = index 
      index += 1
    for token, count in top_title_tokens:
      self.ngram_title_features[token] = index
      index += 1
    self.num_ngram_features = index


  def compute_mi(self, token, word_counts, doc_counts, num_tokens, labels): 
    mi = 0.0
    token_prob = float(sum(word_counts[token].values()))/num_tokens
    num_docs = len(labels)
    for c in [0, 1]: 
      num_of_docs_with_label = len([x for x in labels if x == c])
      class_prob = float(num_of_docs_with_label)/num_docs
      num_of_docs_with_label_and_token = float(doc_counts[c][token]) 
      pos_joint_prob = num_of_docs_with_label_and_token/num_docs 
      pos_denom = class_prob*token_prob
      neg_joint_prob = (num_of_docs_with_label - num_of_docs_with_label_and_token)/num_docs
      neg_denom = class_prob*(1.0 - token_prob) 
      if not pos_joint_prob <= 0.0:
        mi += pos_joint_prob*math.log(pos_joint_prob/pos_denom)
      if not neg_joint_prob <= 0.0:
        mi += neg_joint_prob*math.log(neg_joint_prob/neg_denom)
    return mi


  def select_all_ngrams(self, data):
    # init fetures 
    self.ngram_body_features = {}
    self.ngram_title_features = {}
    index = 1 
    for post in data:
      body_tokens, title_tokens = utils.get_post_tokens(post) 
      # select all
      for token in body_tokens:
        if token not in self.ngram_body_features:
          self.ngram_body_features[token] = index
          index += 1
      for token in title_tokens:
        if token not in self.ngram_title_features:
          self.ngram_title_features[token] = index
          index += 1
    self.num_ngram_features = index
  

##############
# FEAT FUNCS #
##############

  def binary_featurize(self, data): 
    m = len(data) 
    n = self.num_ngram_features
    x = np.zeros((m, n)) 
    for i in range(len(data)):
      post = data[i]
      body_tokens, title_tokens = utils.get_post_tokens(post) 
      for token in body_tokens:
        if token in self.ngram_body_features:
          j = self.ngram_body_features[token]
          x[i][j] = 1.
    return x

  # via http://cs.stanford.edu/~althoff/raop-dataset/altruistic_requests_icwsm.pdf
  # features: gratitude [ ], reciprocity [ ], urgency [ ], status [ X ] 
  def stanford_featurize(self, data):  
    # this could use some engineering to be cleaner 
    # maybe generate some sort of from feature to index 
    # instead of hard coding 
    # TODO add more features
    m = len(data)
    n = 5
    x = np.zeros((m, n))
    for i in range(len(data)):
      post = data[i]
      if "requester_flair_index" in post:
        flair = post["requester_user_flair"] 
        flair_index = utils.get_flair_index(flair) 
      else:
        flair_index = utils.get_flair_index(None)
      account_age_days = post["requester_account_age_in_days_at_request"] 
      req_since_first = post["requester_days_since_first_post_on_raop_at_retrieval"]
      ups_at_ret = post["number_of_upvotes_of_request_at_retrieval"]
      x[i][flair_index] = 1.
      x[i][3] = account_age_days
      x[i][4] = post["number_of_upvotes_of_request_at_retrieval"]

      # These aren't working very well
      #x[i][4] = req_since_first
      #x[i][5] = ups_at_ret

    return x

  def time_featurize(self, data):
    print "time"


  def binary_stanford_featurize(self, data):
    binary = self.binary_featurize(data)
    stanford = self.stanford_featurize(data)
    return np.concatenate((binary, stanford), axis=1)

##########
## MAPS ## 
##########

SELECT_FUNCS = {
  'all': Featurizer.select_all_ngrams,
  'mi': Featurizer.select_top_mi_ngrams
}

FEAT_FUNCS = {
  'binary': Featurizer.binary_featurize, 
  'binary_stanford': Featurizer.binary_stanford_featurize
}
