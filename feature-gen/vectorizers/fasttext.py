import os 
import numpy as np
import json
from sys import stdout

from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from gensim.models import KeyedVectors as kv
import torch
import fastText


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, tokenizer=None, model_file="wiki.en.bin"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model_path = os.path.join(data_dir, "sent_embeddings", "fasttext", model_file)
        if not os.path.isfile(self.model_path):
            print("Couldn't find fasttext .bin file in %s. Exiting" % self.model_path)
            exit(1)
        self.temp_data = os.path.join(data_dir, "store_fasttext")
        if not os.path.exists(self.temp_data):
            os.makedirs(self.temp_data)

    def fit(self, X, y=None):
        print("Loading fasttext model")
        self.trained_model = fastText.load_model(self.model_path)
        return self

    def transform(self, X, y=None):
        sentences = list()
        print("Encoding sentences with FastText")
        for i, sent in enumerate(X):
            stdout.write("\r{:.2%} done".format(float(i) / len(X)))
            stdout.flush()
            sentences.append(list(self.trained_model.get_sentence_vector(sent)))
        return np.array(sentences, dtype='float32')
