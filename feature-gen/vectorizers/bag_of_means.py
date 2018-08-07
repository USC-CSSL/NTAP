### Helper function ###
def load_glove_from_file(fname, vocab=None):
    # vocab is possibly a set of words in the raw text corpus
    if not os.path.isfile(fname):
        raise IOError("You're trying to access a GloVe embeddings file that doesn't exist")
    embeddings = dict()
    with open(fname, 'r') as fo:
        for line in fo:
            tokens = line.split()
            if vocab is not None:
                if tokens[0] not in vocab:
                    continue
            if len(tokens) > 0:
                embeddings[str(tokens[0])] = np.array(tokens[1:], dtype=np.float32)
    return embeddings
### End Helper ###

import sys
sys.path.append("../..")

from set_globals import glove_path, word2vec_path

import os, re, json
import numpy as np
from sys import stdout

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from gensim.models import KeyedVectors as kv
class BoMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_type, tokenizer, ):
        self.embedding_type = embedding_type
        self.tokenizer = tokenizer

        if embedding_type == 'glove':
            self.embeddings_path = glove_path
        else:
            self.embeddings_path = word2vec_path
    
    def get_sentence_avg(self, tokens, embed_size=300, min_threshold=0):
        arrays = list()
        oov = list()
        count = 0
        for token in tokens:
            if self.embedding_type == 'word2vec':
                try:
                    array = self.skipgram_vectors.get_vector(token)
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            elif self.embedding_type == 'glove':
                try:
                    array = self.glove_vectors[token]
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            else:
                raise ValueError("Incorrect embedding_type specified; only possibilities are 'skipgram and 'GloVe'")
        if count <= min_threshold:
            return np.random.rand(embed_size), oov
        sentence = np.array(arrays)
        mean = np.mean(sentence, axis=0)
        return mean, oov

    def fit(self, X, y=None):
        """
        Load selected word embeddings based on specified name 
            (raise exception if not found)
        """

        if not os.path.exists(self.embeddings_path):
            print(self.embeddings_path)
            raise ValueError("Invalid Word2Vec training corpus + method")
        print("Loading embeddings")
        if self.embedding_type == 'word2vec':
            # type is 'gensim.models.keyedvectors.Word2VecKeyedVectors'
            self.skipgram_vectors = kv.load_word2vec_format(self.embeddings_path, binary=True)
        if self.embedding_type == 'glove':
            # type is dict
            self.glove_vectors = load_glove_from_file(self.embeddings_path)
        return self

    def transform(self, raw_docs, y=None):
        avged_docs = list()
        for sentence in raw_docs:
            tokens = self.tokenizer(sentence)
            sentence_mean, out_of_vocabulary = self.get_sentence_avg(tokens)
            avged_docs.append(sentence_mean)
        X = np.array(avged_docs)
        return X
