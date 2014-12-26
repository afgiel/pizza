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
  # features: gratitude [ ], reciprocity [ ], urgency [ ], status [ ] 
  def stanford_featurize(self, data):  
    print "stanford"
    

  def time_featurize(self, data):
    print "time"

##########
## MAPS ## 
##########

SELECT_FUNCS = {
  'all': Featurizer.select_all_ngrams
}

FEAT_FUNCS = {
  'binary': Featurizer.binary_featurize 
}
