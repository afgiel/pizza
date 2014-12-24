from constants import MODELS
import featurizer


class PizzaModel:

  def __init__(self, params):
    self.params = params
    # initialize the model with inverse regularization param C
    self.model = MODELS[params.model](C=params.C)


  def train(self, train_data):
    # initialize featurizer and select features
    f = featurizer.Featurizer(self.params)
    f.select_features(train_data)
    # featurize
    x = f.featurize(train_data)
    y = train_data.labels
    # train
    model.fit(x, y)
    self.f = f

  def test(self, test_data):
    # featurize
    x = self.f.featurize(test_data)
    # predict
    return model.predict(x)

