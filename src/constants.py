
from sklearn.naive_bayes import MultinomialBayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

# map from arg value (string) to feature function 

FEAT_FUNCS = {}
MODELS = {
  "log_res": LogisticRegression,
  "m_nb": MultinomialBayes,
  "b_nb": BernoulliNB
}
