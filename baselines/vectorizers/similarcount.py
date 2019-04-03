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

import re, os
from sys import stdout
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors as kv

# DICT_PATH = os.environ["DICTIONARIES"]

class SimilarCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dictionary_name, embedding_type, tokenizer, dict_path, glove_path, word2vec_path):
        self.embedding_type = embedding_type
        self.dictionary_name = dictionary_name
        self.tokenizer = tokenizer
        dictionary_path = os.path.join(dict_path, dictionary_name + '.dic')
        self.dictionary = dict()
        self.feature_names = dict()
        try:
            with open(dictionary_path, 'r') as dic:
                c = 1
                dic_part = False
                for line in dic:
                    line = line.replace("\n", "").lstrip().rstrip()
                    tokens = line.split()
                    if len(tokens) == 0:
                        continue
                    if c == 1:
                        if line != "%":
                            print("Dictionary format incorrect. Expecting % in the first line.")
                        else:
                            dic_part = True
                    elif dic_part:
                        if line == "%":
                            dic_part = False
                        else:
                            self.feature_names[tokens[0]] = tokens[1]
                    else:
                        num_start = 0
                        key = ""
                        for token in tokens:
                            if not token.isdigit():
                                key += " " + token
                                num_start += 1
                        for token in tokens[num_start:]:
                            self.dictionary.setdefault(self.feature_names[token], []).append(key)
                    c += 1

        except FileNotFoundError:
            print("Could not load dictionary %s from %s" % (self.dictionary_name, dictionary_path))
            exit(1)

        if embedding_type == 'glove':
            self.embeddings_path = glove_path
        else:
            self.embeddings_path = word2vec_path

    def fit(self, X, y=None):
        if not os.path.exists(self.embeddings_path):
            print(self.embeddings_path)
            raise ValueError("Invalid Word2Vec training corpus + method")
        print("Loading embeddings")
        if self.embedding_type == 'word2vec':
            # type is 'gensim.models.keyedvectors.Word2VecKeyedVectors'
            self.embedding_vectors = kv.load_word2vec_format(self.embeddings_path, binary=True)
        if self.embedding_type == 'glove':
            # type is dict
            self.embedding_vectors = load_glove_from_file(self.embeddings_path)

        self.dictionary_embeddings = dict()
        for cat, words in self.dictionary.items():
            self.dictionary_embeddings[cat] = list()
            for word in words:
                try:
                    self.dictionary_embeddings[cat].append(self.embedding_vectors[word.replace("*", "")])
                except Exception:
                    print(word, "does not exist in the embedding file")
        return self

    def transform(self, X, y=None):
        docs = list()
        for sentence in X:
            print(sentence)
            tokens = self.tokenizer(sentence)
            token_embeddings = [self.embedding_vectors[tok.lower()] for tok in tokens
                                if tok in self.embedding_vectors.keys()]
            docs.append(self.similar_count(token_embeddings))
        X = np.array(docs)
        return X


    def similar_count(self, sent_embeddings):
        vector = []
        for cat in sorted(self.dictionary_embeddings.keys()):
            count = 0
            for word in sent_embeddings:
                for keyword in self.dictionary_embeddings[cat]:
                    if self.cosine(word, keyword) > 0.6:
                        count += 1
            vector.append(count)
        return vector

    def cosine(self, a, b):
        return np.dot(a, b)/ (norm(a)* norm(b))


    def get_feature_names(self):
        return sorted(self.dictionary)
