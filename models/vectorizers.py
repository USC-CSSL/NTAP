"""
File: vectorizers.py
Author: Brendan Kennedy
Date: 05/23/2018 - 06/04/2018
Purpose: Implement various data transformers (custom vectorizers as in TfidfVectorizer (sklearn))
"""
from sklearn.feature_extraction.text import CountVectorizer

import os, re
import numpy as np
import json
import subprocess
from sys import stdout

from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from gensim.models import KeyedVectors as kv
import torch
import fastText

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

class DDRVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, embedding_type,
                 tokenizer, dictionary, data_path, similarity):
        self.corpus = training_corpus
        self.embedding_type = embedding_type
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
            tokens = self.tokenizer(sentence)
            sentence_mean, oov = self.get_document_avg(tokens)    
            outputs = [self.similarity(sentence_mean, concept) for concept in concepts]
            ddr_vectors.append(outputs)
        X = np.array(ddr_vectors)
        X = np.nan_to_num(X)
        return X


class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dictionary_name, data_path):
        self.dictionary_name = dictionary_name
        dictionary_path = os.path.join(data_path, 'dictionaries', dictionary_name)
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

    def fit(self, X, y=None):
        self.dictionary_re = dict()
        for cat, words in self.dictionary.items():
            self.dictionary_re[cat] = list()
            for word in words:
                word = word.replace(")", "\\)").replace("(", "\\(").replace(":", "\\:").replace(";", "\\;").replace("/", "\\/")
                if len(word) == 0:
                    continue
                if word[-1] == "*":
                    self.dictionary_re[cat].append(re.compile("(" + word + "\w*)"))
                else:
                    self.dictionary_re[cat].append(re.compile("(" + word + ")"))
        return self

    def transform(self, X, y=None):
        vectors = list()
        c = 0
        for sentence in X:
            c += 1
            stdout.write("\r{:.2%} done".format(float(c) / len(X)))
            stdout.flush()
            vectors.append(self.count_sentence(sentence))
        return np.array(vectors)


    def count_sentence(self, sentence):
        vector = []
        for cat in sorted(self.dictionary_re.keys()):
            count = 0
            for reg in self.dictionary_re[cat]:
                x = len(re.findall(reg, sentence))
                if x > 0:
                    count += x
            if len(sentence.split()) == 0:
                vector.append(0)
            else:
                vector.append(float(count) / float(len(sentence.split())))
        return vector

    def get_feature_names(self):
        return sorted(self.dictionary)
"""
class NgramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n, tokenizer, num_words=10000, stop_words='english'):
        self.n = n
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.stop_words = stop_words

    def fit(self, X, y=None):
        self.count = CountVectorizer(min_df=10,tokenizer=self.tokenizer,
                                        stop_words=self.stop_words,
                                        max_features=10000,
                                        ngram_range=self.n
                                        ).fit(X)
        return self

    def transform(self, X, y=None):
        return np.array(self.count.transform(X).todense())

    def get_feature_names(self):
        ngram_features = list()
        for i in self.n:
            ngram_features.append(self.count.get_feature_names())
        return ngram_features
"""

class LDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, seed, tokenizer, num_topics=100, 
                    num_iter=100, num_words=10000, stop_words='english', ngram=[0, 1]):
        self.num_topics = num_topics
        self.num_iter = num_iter
        self.random_seed = seed
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.stop_words = stop_words
        self.ngram = ngram
    
    def fit(self, X, y=None):
        # find best model by cross-validation (grid-search params)
        self.dt_matrix = CountVectorizer(tokenizer=self.tokenizer,
                                         stop_words=self.stop_words,
                                         max_features=self.num_words, ngram_range=self.ngram).fit_transform(X)

        lda_model = LatentDirichletAllocation(n_components=self.num_topics, learning_method='online',
                                             max_iter=self.num_iter, verbose=1,
                                             random_state=self.random_seed)
        print("Fitting LDA model with 3-fold cross-validation")
        print("Tuning: learning_decay, learning_offset, batch_size")
        """
        choose_lda = GridSearchCV(lda_model, cv=3, iid=True,
                                  param_grid={"learning_decay": np.arange(0.7, 0.9, 0.05),
                                              "learning_offset": np.arange(10, 50, 20),
                                              "batch_size": [32,64,128,256]
                                             })
        """
        self.lda = lda_model.fit(self.dt_matrix)
        print(self.lda.components_)
        return self

    def transform(self, X, y=None):
        return self.lda.transform(self.dt_matrix)

    def get_feature_names(self):
        return ["topic" + str(i) for i in range(self.num_topics)]


class InfersentVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, tokenizer=None, glove_file="glove.840B.300d.txt",
                        model_file="infersent.allnli.pickle"):
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "sent_embeddings", "infersent", model_file)
        self.glove_path = os.path.join(data_dir, "word_embeddings", "GloVe", glove_file)
        if not os.path.isfile(self.glove_path):
            print("Couldn't find GloVe file in %s. Exiting" % self.glove_path)
            exit(1)
        if not os.path.isfile(self.model_path):
            print("Couldn't find infersent model file in %s. Exiting" % self.model_path)
            exit(1)
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        self.model = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        if self.tokenizer is None:
            self.sentences = X
            self.model.set_glove_path(self.glove_path, tokenize=True)
        else:
            self.sentences = [" ".join(self.tokenizer(sent)) for sent in X]
            self.model.set_glove_path(self.glove_path)
            self.model.build_vocab(self.sentences)
        return self

    def transform(self, X, y=None):
        return self.model.encode(self.sentences)

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
