from CSom import *
import math
import numpy as np
import string
import pymongo
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle 
from utils import *
import unicodedata as ud
import scipy

processed_data=load_data('../data_preprocess')

data_unlabeled = [x[:-1] for x in processed_data]

print(data_unlabeled[1])
print("Done Preprocess")


doc_2_vec = TfidfVectorizer(tokenizer=identity_tokenizer, max_features=3000, lowercase=False)
model = CSom(16, data_unlabeled, [0.2, 0.8], 2 ,3000, doc_2_vec)
model.Train()
# PNodes = TfidfVectorizer()
# PNodes = PNodes.fit_transform(corpus_val).todense()

# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(corpus_val[i], PNodes[i])

print("Saving Model...")
ksom_Weights=open('../ksoms.ckpt', 'wb')
model.save(ksom_Weights)
print("Finish Saving Model")

# PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
# PNodes = PNodes.fit_transform(token_text_list).todense()
# sum_quan_err=0
# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     sum_quan_err+=math.sqrt(SuitNode.CalculateDistance(np.squeeze(np.asarray(PNodes[i]))))
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(token_text_list[i], PNodes[i])
# print(f"Quantization Error {sum_quan_err/len(token_text_list)}")
# for iy, ix in np.ndindex(model.m_Som.shape):
#     for i in range(len(model.m_Som[iy,ix].PNodes)):
#         print(iy," ",ix," ",model.m_Som[iy,ix].PNodes[i].corpus)
# model.Plot()