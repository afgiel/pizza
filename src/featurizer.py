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

  def select_all_ngrams(self, data):
    # init fetures 
    self.ngram_features = {}
    index = 1 
    for post in data:
      body_tokens, title_tokens = utils.get_post_tokens(post) 
      all_tokens = body_tokens + title_tokens
      # select all
      for token in all_tokens:
        if token not in self.ngram_features:
          self.ngram_features[token] = index
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
      all_tokens = body_tokens + title_tokens
      for token in all_tokens:
        if token in self.ngram_features:
          j = self.ngram_features[token]
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
    n = 6
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
      x[i][4] = req_since_first
      x[i][5] = req_since_first

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
  'all': Featurizer.select_all_ngrams
}

FEAT_FUNCS = {
  'binary': Featurizer.binary_featurize, 
  'binary_stanford': Featurizer.binary_stanford_featurize
}
