import pymongo
from CSom import *

# client = pymongo.MongoClient("mongodb+srv://fbfighter:fbfighter@fb-topic.ixbkp2u.mongodb.net/?retryWrites=true&w=majority")
#
# db = client["fakenews"]
#
# col = db["fbpost"]
#
# data = col.find()
#
# train_set = []

corpus = ['The sun is the largest celestial body in the solar system',
          'The solar system consists of the sun and eight revolving planets',
          'Ra was the Egyptian Sun God',
          'The Pyramids were the pinnacle of Egyptian architecture',
          'The quick brown fox jumps over the lazy dog']


model = CSom(100, corpus, 10)
model.Train()
