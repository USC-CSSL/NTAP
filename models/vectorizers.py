"""
File: vectorizers.py
Author: Brendan Kennedy
Date: 05/23/2018
Purpose: Implement various data transformers (custom vectorizers as in TfidfVectorizer (sklearn))
"""

from sklearn.base import BaseEstimator, TransformerMixin

class BoMVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, remove_stopwords, preprocessor, tokenizer):
        self.corpus = training_corpus
        self.remove_stopwords = remove_stopwords  # bool
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        # if X is a list of sentences, return a list of averaged word embeddings for word in sentence in list
