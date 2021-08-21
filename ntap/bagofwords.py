import abc
from itertools import zip_longest
import time
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
import tomotopy

from .utils import Preprocessor
from ntap.summarize import _TableMakerMixin
from ntap.validation import _CheckParamsMixin

__all__ = ['DocTerm', 'TFIDF', 'LDA']

logger = logging.getLogger(__name__)

class DocTerm(_CheckParamsMixin, _TableMakerMixin):

    def __init__(self,
                 preprocess='words~links+hashtags+mentions+lowercase',
                 vocab_size=10000, max_df=0.5, min_df=5, **kwargs):

        self.preprocessor = Preprocessor(preprocess)
        self.vocab_size = vocab_size
        self.max_df = max_df
        self.min_df = min_df


        self.N = 0
        self.observed_tokens = 0
        self.vocab = list()

        self.corpus_as_bow = None
        self.corpus_as_csr = None
        self.lengths = list()

        time_now = time.localtime()
        self.created_on = time.strftime("%d %b %Y %H:%M:%S", time_now)

        self._verify_params()

    def top_vocab(self, k=20):
        """ Return top _k_ items in vocabulary (by corpus frequency) """

        word_freq_pairs = [(self.vocab[id_], f) for id_, f in self.vocab.cfs.items()]
        sorted_vocab = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)
        vocab_by_freq, _ = zip(*sorted_vocab)
        return [str(a) for a in list(vocab_by_freq)[:k]]

    def _make_main_table_elements(self):

        data = {
            'Object Type': ['DocTerm'],
            'Created': [self.created_on] if self.created_on else ['---'],
            'Preprocess': [self.preprocessor.description]
        }

        return data, None

    def _make_content_table_elements(self):

        if self.is_fit():
            n, k = self.corpus_as_csr.shape
            nnz = self.corpus_as_csr.nnz
            size = n * k
            nnz_prop = (size - nnz) / size
        else:
            n, k, nnz, size = 0, 0, 0, 0

        data = {
            f'{self.__class__.__name__} Matrix': [f"(documents: {self.N}, terms: {len(self.vocab)})"],
            'Non-/sparse entries': [f"{nnz:,}/{size:,}"] if self.N else ['---'],
            'Sparsity': [f"{nnz_prop:.2%}"] if self.N else ['---'],
        }

        title = None

        return data, title


    def fit(self, corpus):

        self._verify_corpus(corpus)
        self.N = len(corpus)

        tokens = self.preprocessor.transform(corpus)
        self.observed_tokens = tokens.apply(len).sum()

        vocab = Dictionary(tokens)
        vocab.filter_extremes(no_above=self.max_df,
                              no_below=self.min_df,
                              keep_n=self.vocab_size)
        vocab.compactify()

        self.vocab = vocab
        self.corpus_as_tokens = tokens
        self.corpus_as_bow = [self.vocab.doc2bow(doc) for doc in tokens]
        self.corpus_as_csr = corpus2csc(self.corpus_as_bow,
                                        num_terms=len(self.vocab)).T

        self.lengths = [len(d) for d in self.corpus_as_bow]
        self.num_empty_docs = self.lengths.count(0)

        time_now = time.localtime()
        self.created_on = time.strftime("%d %b %Y %H:%M:%S", time_now)

        return self

    def is_fit(self):
        """ Return True if DocTerm has been fit to a corpus """

        return (self.corpus_as_bow is not None
                or self.corpus_as_csr is not None)

    def transform(self, corpus):
        """ Apply vocab to new corpus, producing csr matrix """

        self._verify_corpus(corpus)

        if not self.is_fit():
            self.fit(corpus)

        tokens = self.preprocessor.transform(corpus)
        bow = [self.vocab.doc2bow(doc) for doc in tokens]
        csr_corpus = corpus2csc(bow, num_terms=len(self.vocab)).T

        return csr_corpus

    def _terms_covered_by_vocab(self, terms):
        """ Terms has words and stems ([\w]+\*) """

        _types = set(self.vocab.token2id.keys())
        matches = set()
        for t in set(terms):
            if not t.endswith('*'):
                if t in _types:
                    matches.add(t)
            else:
                stem = t.strip('*')
                for _type_i in _types:
                    if _type_i.startswith(stem):
                        matches.add(t)
                        continue
        return matches

    def __len__(self):
        return self.N

class TFIDF(DocTerm):

    def __init__(self, preprocessor='stopwords~links+hashtags+mentions+lowercase',
                 vocab_size=10000, max_df=0.5, min_df=5, **kwargs):
        super().__init__(preprocessor=preprocessor,
                         vocab_size=vocab_size,
                         max_df=max_df, min_df=min_df)

        # TODO: make tfidf args explicit
        self.__dict__.update(kwargs)

        self.tfidf_model = None

    def fit(self, corpus):

        docterm = super().fit(corpus)

        self.tfidf_model = TfidfModel(self.corpus_as_bow, 
                                      id2word=self.vocab)
        self.corpus_as_tfidf = self.transform(corpus)
        return self

    def is_fit(self):
        return super().is_fit() and self.tfidf_model is not None

    def transform(self, corpus):

        if not self.is_fit():
            self.fit(corpus)

        tokens = self.preprocessor.transform(corpus)
        bow = [self.vocab.doc2bow(doc) for doc in tokens]
        docs = [self.tfidf_model.__getitem__(doc) for doc in bow]

        return corpus2csc(docs, num_terms=len(self.vocab)).T

    def _make_main_table_elements(self):
        data, title = super()._make_main_table_elements()
        data['Weighting'] = ['TF-IDF']

        return data, title

class TopicModel(_CheckParamsMixin, _TableMakerMixin):

    model_list = {
        'lda': tomotopy.LDAModel,
        'hdp': tomotopy.HDPModel
    }


    def __init__(self, method='lda', 
                 preprocessor='stopwords+collocations~social+lowercase',
                 k=50, iters=50, **kwargs):
        #super().__init__(preprocessor=preprocessor, **kwargs)

        self.method = method
        self.k = k
        self.iters = iters

        if self.method in self.model_list:
            self.mdl_class = self.model_list[self.method]
        else:
            logger.warn(f"Could not find model {self.method}. Options:\n"
                        ' '.join(self.model_list.keys()))

        self.docterm = DocTerm(preprocessor=preprocessor, **kwargs)

    def __load_docs(self):
        """ Returns na_mask and list of documents """

        na_mask = [len(doc) > 0 for doc in self.docterm.corpus_as_tokens]
        for doc in self.docterm.corpus_as_tokens.values:
            if len(doc) > 0:
                self.mdl.add_doc(doc)
        return na_mask

    def fit(self, corpus):

        if not self.docterm.is_fit():
            self.docterm.fit(corpus)

        self.mdl = self.mdl_class(k=self.k)
        self.na_mask = self.__load_docs()

        for i in range(0, self.iters, 10):
            self.mdl.train(i)
            logger.info(f'Iteration: {i}\tLog-likelihood: {self.mdl.ll_per_word}')

        time_now = time.localtime()
        self.train_date = time.strftime("%d %b %Y %H:%M:%S", time_now)

        #self.print_topics()
        return self

    def _make_main_table_elements(self):
        prev, _ = self.docterm._make_main_table_elements()
        title = f'{self.__class__.__module__}.{self.__class__.__name__}'

        data = dict(
            Model=[f'{self.mdl.__class__.__module__}.{self.mdl.__class__.__name__}'],
            Implementation=['Collapsed Gibbs Sampling'],
            Date=[self.train_date],
            Log_Likelihood=[f'{self.mdl.ll_per_word:.3f}'],
            #Perplexity=[f'{self.mdl.perplexity:.3f}'],
            Documents=[f'{len(self.mdl.docs)}'],
            Preprocess=prev['Preprocess']
        )

        return data, title

    def print_topics(self):

        self.mdl.summary(initial_hp=False, params=False, topic_word_top_n=15)

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

