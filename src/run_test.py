import pizza_model
import constants
from constants import MODELS
from constants import FEAT_FUNCS
import argparse


def get_args():
  parser = argparse.ArgumentParser(description="Test different learning models, feature representations and parameters on the reddit datasets.")
  parser.add_argument('-m', '-model', choices=MODELS.keys(), default='log_res', type=str)

  args = parser.args
  return args 

params = get_args()
model = pizza_model.PizzaModel(params)

