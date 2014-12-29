from constants import MODELS
import featurizer
import utils
from sklearn.metrics import classification_report


class PizzaModel:

  def __init__(self, params):
    self.params = params
    # initialize the model with inverse regularization param C
    if params.model == "log_res":
      self.model = MODELS[params.model](C=params.C)
    else:
       self.model = MODELS[params.model]()


  def train(self, train_data):
    f = featurizer.Featurizer(self.params)
    print "SELECTING FEATURES"
    f.select_features(train_data)
    print "FEATURIZING TRAIN SET"
    x = f.featurize(train_data)
    y = utils.get_labels_from_post_list(train_data) 
    print "TRAINING MODEL"
    self.model.fit(x, y)
    self.f = f
    print "TRAINING CLASSIFICATION REPORT"
    test_classification = self.model.predict(x)
    print classification_report(test_classification, y)

  def test(self, test_data):
    print "FEATURIZING TEST SET"
    x = self.f.featurize(test_data)
    print "PREDICTING"
    return self.model.predict(x)

