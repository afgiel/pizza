import json

class Pizza_Post():

   def __init__(self, data):
      self.data = data
      self.features = []

   def add_feature(self, key, value):
       self.features[key] = value

   def get_feature(self, key):
      return self.features[key]
