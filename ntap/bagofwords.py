import abc
import json
import re
import os
import logging
import subprocess
from collections import Counter
from typing import Iterable, Union

# 3rd party imports
import liwc
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
import tomotopy as tpy

from ntap.parse import Preprocessor, Tokenizer

__all__ = ['DocTerm', 'TFIDF', 'LDA']

logger = logging.getLogger(__name__)

def _verify_corpus(corpus):

    assert isinstance(corpus, (list, pd.Series, np.ndarray)), "corpus must be list-like"

    if isinstance(corpus, (list, np.ndarray)):
        assert len(corpus) == sum([isinstance(a, str) for a in corpus]), "corpus must have strings"
    elif isinstance(corpus, pd.Series):
        assert len(corpus) == sum([isinstance(a, str) for a in corpus.values]), "corpus must have strings"

def _verify_params(tokenizer, vocab_size, max_df):
    assert isinstance(vocab_size, int), "vocab_size must be an integer"
    assert max_df >= 0. and max_df <= 1, "max_df must be between 0 and 1"

    tokenizer_options = ", ".join(_TOKENIZERS.keys())
    assert tokenizer in _TOKENIZERS, "valid tokenizers: {}".format(tokenizer_options)

class DocTerm:

    def __init__(self, tokenizer='word', preprocessor='all', 
                 vocab_size=10000, max_df=0.5, **kwargs):

        #_verify_params(tokenizer, vocab_size, max_df)
        self.tokenizer = Tokenizer(tokenizer)
        self.preprocessor = Preprocessor(preprocessor)
        self.vocab_size = vocab_size
        self.max_df = max_df

    def top_vocab(self, k=20):
        """ Return top _k_ items in vocabulary (by corpus frequency)

        pre: self.vocab exists (by calling self.fit(...))

        """

        assert hasattr(self, 'vocab'), "No vocab object found; fit DocTerm to corpus"

        word_freq_pairs = [(self.vocab[id_], f) for id_, f in self.vocab.cfs.items()]
        sorted_vocab = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)
        vocab_by_freq, _ = zip(*sorted_vocab)
        return [str(a) for a in list(vocab_by_freq)[:k]]

    def fit(self, corpus):

        _verify_corpus(corpus)
        self.N = len(corpus)

        cleaned = self.preprocessor.transform(corpus)
        tokens = self.tokenizer.transform(cleaned)

        vocab = Dictionary(tokens)
        vocab.filter_extremes(no_above=self.max_df, keep_n=self.vocab_size)
        vocab.compactify()
        self.vocab = vocab

        self.bow = [self.vocab.doc2bow(doc) for doc in tokens]
        self.lengths = [len(d) for d in self.bow]

    def transform(self, corpus):

        _verify_corpus(corpus)

        if 'bow' not in self.__dict__:
            self.fit(corpus)
            bow = self.bow
        else:
            cleaned = self.preprocessor.transform(corpus)
            tokens = self.tokenizer.transform(cleaned)
            bow = [self.vocab.doc2bow(doc) for doc in tokens]

        return corpus2csc(bow, num_terms=len(self.vocab)).T

    def __len__(self):
        return self.N

    def __str__(self):
        min_length = min(self.lengths)
        max_length = max(self.lengths)
        median_length = np.median(self.lengths)
        mean_length = np.mean(self.lengths)
        return ("DocTerm Object (documents: {}, terms: {})\n"
                "Doc Lengths ({}-{}): mean {:.2f}, median {:.2f}\n"
                "Top Terms: {}") \
            .format(self.N, len(self.vocab), min_length, max_length,
                    median_length, mean_length,
                    " ".join(self.top_vocab()))


class TFIDF(DocTerm):

    def __init__(self, tokenizer='word', preprocessor='all',
                 vocab_size=10000, max_df=0.5, **kwargs):
        super().__init__(tokenizer=tokenizer, preprocessor=preprocessor,
                         vocab_size=vocab_size, max_df=max_df)

        # TODO: make tfidf args explicit
        self.__dict__.update(kwargs)


    def fit(self, corpus):

        _verify_corpus(corpus)
        self.N = len(corpus)

        cleaned = self.preprocessor.transform(corpus)
        tokens = self.tokenizer.transform(cleaned)

        vocab = Dictionary(tokens)
        vocab.filter_extremes(no_above=self.max_df, keep_n=self.vocab_size)
        vocab.compactify()
        self.vocab = vocab

        self.bow = [self.vocab.doc2bow(doc) for doc in tokens]
        self.lengths = [len(d) for d in self.bow]

        self.tfidf_model = TfidfModel(self.bow, id2word=self.vocab)

        return self

    def transform(self, corpus):

        _verify_corpus(corpus)

        if 'tfidf_model' not in self.__dict__:
            self.fit(corpus)
            bow = self.bow
        else:
            cleaned = self.preprocessor.transform(corpus)
            tokens = self.tokenizer.transform(cleaned)
            bow = [self.vocab.doc2bow(doc) for doc in tokens]

        docs = [self.tfidf_model.__getitem__(doc) for doc in bow]

        return corpus2csc(docs, num_terms=len(self.vocab)).T

class LDA(DocTerm):
    def __init__(self, method='vanilla', k=50, num_iterations=50,
                 tokenizer='regex', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.k = k
        self.num_iterations = num_iterations


    def __load_docs(self, 
                    data: Union[str, Iterable[str]]):
        """ Returns na_mask and list of documents """

        cleaned = self.preprocessor.transform(data)
        tokens = self.tokenizer.transform(cleaned)

        self.curr_corpus = tokens

        na_mask = [len(doc) > 0 for doc in tokens]
        for doc in tokens.values:
            if len(doc) > 0:
                self.mdl.add_doc(doc)
        return na_mask

    def fit(self, corpus):

        self.mdl = tpy.LDAModel(k=self.k)
        self.na_mask = self.__load_docs(corpus)

        for i in range(0, self.num_iterations, 10):
            self.mdl.train(i)
            logger.info(f'Iteration: {i}\tLog-likelihood: {self.mdl.ll_per_word}')

        return self

    def print_topics(self):

        for k in range(self.mdl.k):
            print('Top 10 words of topic #{}'.format(k))
            print(self.mdl.get_topic_words(k, top_n=10))

    def transform(self, data=None, return_training_docs=True):

        if not return_training_docs:
            assert data is not None, "Missing new data argument"

            na_mask = self.__load_docs(data)
            infer_docs = [None] * sum(na_mask)
            i = 0
            for doc in tokens:
                if len(doc) > 0:
                    infer_docs[i] = self.mdl.make_doc(words=doc)
                    i += 1

            logger.info("Inferring doc probabilities for LDA")
            dists, lls = self.mdl.infer(infer_docs)

        else:  # extract topic dist from model object
            dists = np.zeros((len(self.mdl.docs), self.k))
            for i, doc in enumerate(self.mdl.docs):
                dists[i, :] = np.array(doc.get_topic_dist())
            na_mask = self.na_mask
            data = self.curr_corpus

        stitched_docs = np.zeros((len(data), self.k))
        inferred_doc_index = 0
        for i in range(len(data)):
            not_na = na_mask[i]
            if not_na:
                stitched_docs[i, :] = np.array(dists[inferred_doc_index])
                inferred_doc_index += 1
            else:
                stitched_docs[i, :] = np.NaN

        return stitched_docs

