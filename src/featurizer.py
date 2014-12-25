import numpy as np

SELECT_FUNCS = {
  'all': Featurizer.select_all_ngrams
}

FEAT_FUNCS = {
  'binary': Featurizer.binary_featurize 
}

class Featurizer():

  def __init__(self, params):
    self.params = params

  def select_features(self, data):
    self.feature_map = SELECT_FUNCS[self.params.selector](data) 

  def featurize(self, data):
    return FEAT_FUNCS[self.params.featurizer](data)

################
# SELECT FUNCS #
################

  def select_all_ngrams(self, data):


##############
# FEAT FUNCS #
##############

  def binary_featurize(self, data): 


  
