import argparse
import json
import random
from sklearn.metrics import classification_report

import constants
from constants import MODELS
import featurizer
from featurizer import SELECT_FUNCS, FEAT_FUNCS
import pizza_model
import utils


def get_args():
  parser = argparse.ArgumentParser(description="Test different learning models, feature representations and parameters on the reddit datasets.")
  parser.add_argument('-model', '-m', choices=MODELS.keys(), default='log_res', type=str)
  parser.add_argument('-C', '-c', default=1.0, type=float)
  parser.add_argument('-selector', '-s', choices=SELECT_FUNCS.keys(), default='all', type=str)
  parser.add_argument('-featurizer', '-f', choices=FEAT_FUNCS.keys(), default='binary', type=str)
  parser.add_argument('-num_body_tokens', '-nbt', default=5000, type=int)
  parser.add_argument('-num_title_tokens', '-ntt', default=3000, type=int)
  parser.add_argument('--TESTING', action='store_true', default=False)
  args = parser.parse_args()
  return args 


# get experiment params
params = get_args()

# load data into memory
# TODO
print "LOADING DATA"
json_data = open('../data/train.json')
train_data = json.load(json_data)
json_data.close()
test_data = [] 
# test or dev
if params.TESTING:
  json_data = open('../data/test.json')
  test_data = json.load(json_data)
  json_data.close()
else: 
  test_data = random.sample(train_data, int(len(train_data)*.1)) 
  train_data = [post for post in train_data if post not in test_data] 

# intialize model
model = pizza_model.PizzaModel(params)

# train 
model.train(train_data)

# test 
predictions = model.test(test_data)

# evaluate
if not params.TESTING:
  desired = utils.get_labels_from_post_list(test_data)
  print classification_report(desired, predictions)
else:
  # write output to file
  utils.write_output(test_data, predictions) 
