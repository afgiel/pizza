import pizza_model
import constants
from constants import MODELS
import featurizer
from featurizer import SELECT_FUNCS, FEAT_FUNCS
import argparse
import json


def get_args():
  parser = argparse.ArgumentParser(description="Test different learning models, feature representations and parameters on the reddit datasets.")
  parser.add_argument('-model', '-m', choices=MODELS.keys(), default='log_res', type=str)
  parser.add_argument('-C', '-c', default=1.0, type=float)
  parser.add_argument('-selector', '-s', choices=SELECT_FUNCS.keys(), default='all', type=str)
  parser.add_argument('-featurizer', '-f', choices=FEAT_FUNCS.keys(), default='binary', type=str)
  args = parser.args
  return args 


# get experiment params
params = get_args()

# load data into memory
# TODO
json_data = open('../data/train.json')
train_data = json.load(json_data)
json_data.close()
json_data = open('../data/test.json')
test_data = json.load(json_data)
json_data.close()

# intialize model
model = pizza_model.PizzaModel(params)

# train 
model.train(train_data)

# test 
predictions = model.test(test_data)

#evaluate
# TODO

# write output to file
# TODO 
