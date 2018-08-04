
import os, re, json
import numpy as np
from sys import stdout

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from gensim.models import KeyedVectors as kv
class BoMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, embedding_type, tokenizer, data_path):
        self.corpus = training_corpus
        self.embedding_type = embedding_type
        self.tokenizer = tokenizer
        self.data_path = os.path.join(data_path, 'word_embeddings') 

        self.corpus_filenames = {'GoogleNews': 'GoogleNews-vectors-negative300.bin',
                                 'common_crawl': 'glove.42B.300d.txt',
                                 'wiki_gigaword': 'glove.6B.300d.txt'}
    
    def get_sentence_avg(self, tokens, embed_size=300, min_threshold=0):
        arrays = list()
        oov = list()
        count = 0
        for token in tokens:
            if self.embedding_type == 'skipgram':
                try:
                    array = self.skipgram_vectors.get_vector(token)
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            elif self.embedding_type == 'GloVe':
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

        self.embeddings_path = os.path.join(self.data_path, self.embedding_type, 
                                            self.corpus_filenames[self.corpus])
        if not os.path.exists(self.embeddings_path):
            print(self.embeddings_path)
            raise ValueError("Invalid Word2Vec training corpus + method")
        print("Loading embeddings")
        if self.embedding_type == 'skipgram':
            # type is 'gensim.models.keyedvectors.Word2VecKeyedVectors'
            self.skipgram_vectors = kv.load_word2vec_format(self.embeddings_path, binary=True)
        if self.embedding_type == 'GloVe':
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
        print(X.shape)
        return X
