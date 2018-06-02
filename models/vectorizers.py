"""
File: vectorizers.py
Author: Brendan Kennedy
Date: 05/23/2018 - 05/30/2018
Purpose: Implement various data transformers (custom vectorizers as in TfidfVectorizer (sklearn))
"""

import os
import numpy as np
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


from gensim.models import KeyedVectors as kv

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

class BoMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, embedding_type, remove_stopwords, 
                 preprocessor, tokenizer, data_path):
        self.corpus = training_corpus
        self.embedding_type = embedding_type
        self.remove_stopwords = remove_stopwords  # bool
        self.preprocessor = preprocessor
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
            return np.rand(embed_size), oov
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
            processed = self.preprocessor(sentence)
            tokens = self.tokenizer(processed)
            sentence_mean, out_of_vocabulary = self.get_sentence_avg(tokens)
            avged_docs.append(sentence_mean)
        X = np.array(avged_docs)
        print(X.shape)
        return X

class DDRVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, embedding_type, remove_stopwords, 
                 preprocessor, tokenizer, dictionary, data_path, similarity):
        self.corpus = training_corpus
        self.embedding_type = embedding_type
        self.remove_stopwords = remove_stopwords  # bool
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.corpus_path = os.path.join(data_path, 'word_embeddings') 
        dictionary_path = os.path.join(data_path, 'dictionaries', dictionary + '.json')
        try:
            with open(dictionary_path, 'r') as fo:
                data = json.load(fo)
                self.dictionary = data.items()
        except FileNotFoundError:
            print("Could not load dictionary %s from %s" % (self.dictionary, dictionary_path))
            exit(1)

        self.similarity = similarity
        self.corpus_filenames = {'GoogleNews': 'GoogleNews-vectors-negative300.bin',
                                 'common_crawl': 'glove.42B.300d.txt',
                                 'wiki_gigaword': 'glove.6B.300d.txt'}

    def get_feature_names(self):
        return [item[0] for item in self.dictionary]

    def get_document_avg(self, tokens, embed_size=300, min_threshold=0):
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
            return np.zeros(embed_size), oov
        sentence = np.array(arrays)
        mean = np.mean(sentence, axis=0)
        return mean, oov

    def fit(self, X, y=None):
        """
        Load selected word embeddings based on specified name 
            (raise exception if not found)
        """

        self.embeddings_path = os.path.join(self.corpus_path, self.embedding_type, 
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
        print("Calculating dictionary centers")
        concepts = list()
        for concept, words in self.dictionary:
            concept_mean, _ = self.get_document_avg(words)
            concepts.append(concept_mean)
        ddr_vectors = list()
        for sentence in raw_docs:
            processed = self.preprocessor(sentence)
            tokens = self.tokenizer(processed)
            sentence_mean, oov = self.get_document_avg(tokens)    
            outputs = [self.similarity(sentence_mean, concept) for concept in concepts]
            ddr_vectors.append(outputs)
        X = np.array(ddr_vectors)
        X = np.nan_to_num(X)
        return X

class LDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, seed, tokenizer, preprocessor, num_topics=100, 
                    num_iter=10, num_words=10000, stop_words='english'):
        self.num_topics = num_topics
        self.num_iter = num_iter
        self.random_seed = seed
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.num_words = num_words
        self.stop_words = stop_words
    
    def fit(self, X, y=None):
        # find best model by cross-validation (grid-search params)
        self.dt_matrix = CountVectorizer(preprocessor=self.preprocessor,
                                         tokenizer=self.tokenizer,
                                         stop_words=self.stop_words,
                                         max_features=self.num_words).fit_transform(X)

        lda_model = LatentDirichletAllocation(n_components=self.num_topics, learning_method='online',
                                             max_iter=self.num_iter, verbose=1,
                                             random_state=self.random_seed)
        print("Fitting LDA model with 3-fold cross-validation")
        print("Tuning: learning_decay, learning_offset, batch_size")
        choose_lda = GridSearchCV(lda_model, cv=3, iid=True,
                                  param_grid={"learning_decay": np.arange(0.7, 0.9, 0.05),
                                              "learning_offset": np.arange(10, 50, 20),
                                              "batch_size": [32,64,128,256]
                                             })
        choose_lda.fit(self.dt_matrix)

        self.lda = choose_lda.best_estimator_
        return self

    def transform(self, X, y=None):
        return self.lda.transform(self.dt_matrix)

    def get_feature_names(self):
        return ["topic" + str(i) for i in range(self.num_topics)]
