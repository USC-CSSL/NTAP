import abc
import json
import re
import os
import logging
import subprocess
from collections import Counter

# 3rd party imports
import liwc
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary as DictGensim
from gensim.models.wrappers import LdaMallet
from gensim.models import LdaModel, TfidfModel
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.matutils import corpus2csc
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix

_WORD_RE = re.compile(r'[\w\_]{2,20}')
_TOKENIZERS = {'regex': lambda x: _WORD_RE.findall(x),
              'whitespace': lambda x: x.split()}

__all__ = ['DocTerm', 'TFIDF', 'LDA', 'Dictionary']

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

    def __init__(self, tokenizer='regex', vocab_size=10000, max_df=0.5, **kwargs):
        """ corpus is list-like; contains str documents """

        _verify_params(tokenizer, vocab_size, max_df)
        self.tokenizer = _TOKENIZERS[tokenizer]
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


    def get_tokenized(self, corpus):
        if isinstance(corpus, pd.Series):
            return corpus.apply(self.tokenizer)
        else:
            return [self.tokenizer(doc) for doc in corpus]

    def fit(self, corpus):

        _verify_corpus(corpus)
        tokens = self.get_tokenized(corpus)
        vocab = DictGensim(tokens)
        vocab.filter_extremes(no_above=self.max_df, keep_n=self.vocab_size)
        vocab.compactify()

        self.vocab = vocab
        self.tokenized = tokens
        self.bow = [self.vocab.doc2bow(doc) for doc in self.tokenized]

        self.lengths = [len(d) for d in self.bow]
        self.N = len(self.bow)

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


class TFIDF:
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

    def transform(self, corpus):

        if not isinstance(corpus, DocTerm):

            _verify_corpus(corpus)
            docterm_params = {k: v for k, v in self.__dict__.items() 
                              if k in {'vocab_size', 'max_df'}}
            dt = DocTerm(**docterm_params)
        else:
            dt = corpus

        if not hasattr(dt, 'bow'):  # not fit object
            dt.fit(corpus)

        self.vocab = dt.vocab

        self.model = TfidfModel(dt.bow, id2word=self.vocab)
        docs = [self.model.__getitem__(doc) for doc in dt.bow]
        return corpus2csc(docs, num_terms=len(self.vocab)).T


class LDA:
    def __init__(self, method='online', num_topics=50, num_iterations=500,
                 optimize_interval=10, tokenizer='regex', **kwargs):
        self.__dict__.update(kwargs)
        self.method = method
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.optimize_interval = optimize_interval  # hyperparameters

        self.tokenizer = _TOKENIZERS[tokenizer]
        #self.model = self.__fit_lda_model()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if type(model) == str:
            if os.path.exists(model):
                self._model = utils.SaveLoad.load(model)
        else:
            self._model = model

    def get_docterm_params(self):
        """ Return param dict for docterm param set """

        valid_params = {'vocab_size', 'max_df'}
        return {k: v for k, v in self.__dict__.items() if k in valid_params}

    def fit(self, corpus):

        if not isinstance(corpus, DocTerm):

            _verify_corpus(corpus)
            docterm_params = self.get_docterm_params()

            dt = DocTerm(**docterm_params)
        else:
            dt = corpus

        if not hasattr(dt, 'bow'):  # not fit object
            dt.fit(corpus)

        self.vocab = dt.vocab

        if self.method == 'online':
            model = LdaModel(corpus=dt.bow,
                             num_topics=self.num_topics,
                             iterations=self.num_iterations,
                             id2word=dt.vocab)

        elif self.method == 'gibbs':
            if 'mallet_path' not in self.__dict__:
                raise ValueError("Cannot gibbs sampling without setting \'mallet_path\'")
            try:
                model = LdaMallet(mallet_path=self.mallet_path,
                                  #prefix=prefix,
                                  corpus=dt.bow,
                                  id2word=dt.vocab,
                                  iterations=self.num_iterations,
                                  num_topics=self.num_topics,
                                  optimize_interval=self.optimize_interval)
            except subprocess.CalledProcessError:
                raise ValueError(f'Bad mallet_path argument ({self.mallet_path})')
            model = malletmodel2ldamodel(model)
        else:
            raise ValueError("Invalid LDA method: {}".format(self.method))
        self.model = model

        return self

    def transform(self, corpus):
        if isinstance(corpus, pd.Series):
            tokenized = corpus.apply(self.tokenizer)
            dt = tokenized.apply(lambda tokens: self.model[self.vocab.doc2bow(tokens)])
            return corpus2csc(dt.values.tolist(), 
                              num_terms=self.num_topics, 
                              num_docs=len(dt)).T
        else:
            tokenized = [self.tokenizer(doc) for doc in corpus]
            bow = [self.vocab.doc2bow(doc) for doc in tokenized]
            return [self.model[doc] for doc in bow]


class Dictionary(DocTerm):

    def __init__(self, dic_path):
        super().__init__()

        self.dic_path = dic_path
        if not os.path.exists(dic_path):
            logger.exception(f'Could not load .dic file: {dic_path}')

        self.dic_parser, self.names = liwc.load_token_parser(dic_path)
        #self.dic_items = [[ngram.replace(' ', '_') for ngram in l] for l in self.dic_items]

    def transform(self, corpus):

        _verify_corpus(corpus)

        self.tokenized = self.get_tokenized(corpus)
        dic_docs = list()
        self.lengths = list()
        for doc in self.tokenized:
            counts = dict(Counter(cat for token in doc for cat in self.dic_parser(token)))
            N = len(doc)
            for liwc_cat in self.names:
                if liwc_cat not in counts:
                    counts[liwc_cat]= 0
                if N > 0:
                    counts[liwc_cat] /= N
            dic_docs.append(counts)
            self.lengths.append(N)
        dic_docs = pd.DataFrame(dic_docs)
        self.names = list(dic_docs.columns)
        return csr_matrix(dic_docs.values)

