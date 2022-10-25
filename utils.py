import pickle
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from PNode import *
from CSom import *



def identity_tokenizer(text):
    return text
def load_model(filepath):
    with open(filepath, 'rb') as inp:
        model=pickle.load(inp)
    return model

def load_data(filepath):
    with open(filepath, 'rb') as inp:
        data=pickle.load(inp)
    return data

def create_pnode(model, pre_data):
    corpus = [x[0] for x in pre_data]
    writing_style = [x[1] for x in pre_data]
    # print(corpus[0])
    # exit()
    # PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    PNodes_content_endcode = model.doc_2_vectorizer.transform(corpus).todense()
    PNodes_writingstyle_encode = np.array(writing_style)
    PNodes=np.asarray([PNode(corpus=corpus[i], num_component=2, vectors=[np.squeeze(np.asarray(PNodes_content_endcode[i])), PNodes_writingstyle_encode[i]]) for i in range(len(corpus))])
    return PNodes