import pymongo
from CSom import *
import math
import numpy as np
import string
import pymongo
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import unicodedata as ud

client = pymongo.MongoClient("mongodb+srv://fbfighter:fbfighter@fb-topic.ixbkp2u.mongodb.net/?retryWrites=true&w=majority")

db = client["fakenews"]

col = db["fbpost"]

data = col.find()

corpus = []

letters = string.ascii_lowercase

print("Start preprocess")
for x in data:
    corpus.append(x['text'])

def preprocess_text(text):
    special_char = ['̼', '“', '”', '–']
    punctualtion_list = string.punctuation + "".join(special_char)
    removed_punctuation = "".join([i for i in text if i not in punctualtion_list])
    lower_case = removed_punctuation.lower()
    # translated_text = lower_case.translate(translator)
    # return translated_text
    
    return lower_case

# for i in clean_text:
#     print("post-----------")
#     print(i)

def find_special_char(text):
    regex = "[^a-z0-9A-Z_\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]"
    set_char = set(re.findall(regex, text))
    return set_char

def replace_special_char(text):
    mapping_list = {}
    set_char = find_special_char(text)
    for char in set_char:
        try:
            char_name = ud.name(char)
            letter = char_name.split(' ')[-1].lower()
            if letter in string.ascii_lowercase:
                mapping_list[ord(char)] = letter
        except:
            pass
    return text.translate(str.maketrans(mapping_list))

clean_text = list(map(preprocess_text, corpus))
clean_text2 = list(map(replace_special_char, clean_text))



def tokenization(text):
    tokens = re.split('\s+', text)
    return tokens

token_text_list = list(map(tokenization, clean_text2))

print("Done Preprocess")

def identity_tokenizer(text):
    return text


model = CSom(26, token_text_list, 500)
model.Train()
# PNodes = TfidfVectorizer()
# PNodes = PNodes.fit_transform(corpus_val).todense()

# for i in range(PNodes.shape[0]):
#     SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
#     # print(PNodes[i])
#     # print(SuitNode)
#     SuitNode.addPNode(corpus_val[i], PNodes[i])

PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
PNodes = PNodes.fit_transform(token_text_list).todense()
sum_quan_err=0
for i in range(PNodes.shape[0]):
    SuitNode, _, _ = model.FindBestMatchingNode(PNodes[i])
    sum_quan_err+=math.sqrt(SuitNode.CalculateDistance(np.squeeze(np.asarray(PNodes[i]))))
    # print(PNodes[i])
    # print(SuitNode)
    SuitNode.addPNode(token_text_list[i], PNodes[i])
print(f"Quantization Error {sum_quan_err/len(token_text_list)}")
# for iy, ix in np.ndindex(model.m_Som.shape):
#     for i in range(len(model.m_Som[iy,ix].PNodes)):
#         print(iy," ",ix," ",model.m_Som[iy,ix].PNodes[i].corpus)
# model.Plot()