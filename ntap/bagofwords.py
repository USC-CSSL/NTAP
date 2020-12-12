import abc
import json
import re
import os

# 3rd party imports
import numpy as np
import pandas as pd
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary as DictGensim
from gensim.matutils import corpus2csc
from scipy.spatial.distance import cosine

# local imports
from .utils import read_dictionary

_WORD_RE = re.compile(r'[\w\_]{2,20}')



class Dictionary:

    def __init__(self, dic_path=None, dic_data=None):

        if dic_path is not None:
            with open(dic_path) as liwc_file:
                self.categories, self.dic_items = read_dictionary(liwc_file)
        elif dic_data is not None:
            self.categories, self.dic_items = dic_data
        else:
            raise ValueError("No path to dictionary/Dictionary object provided")

        self.dic_items = [[ngram.replace(' ', '_') for ngram in l] for l in self.dic_items]
        #self.dic = {c: items for c, items in zip(self.categories, self.dic_items)}
        self.__build_regexes()

    def __build_regexes(self):
        """ Initialize regexes for word searching """
        return

class DocTerm:

    tokenizers = {'regex': lambda x: _WORD_RE.findall(x),
                  'basic': lambda x: x.split()}

    def __init__(self, corpus, tokenizer='regex', vocab_size=10000, 
                 max_df=0.5, lang='english', **kwargs):
        """ corpus is list-like; contains str documents """

        assert isinstance(corpus, list) or isinstance(corpus, pd.Series) \
            or isinstance(corpus, np.array), "corpus of DocTerm must be list-like"

        if isinstance(corpus, list) or isinstance(corpus, np.array):
            assert len(corpus) == sum([isinstance(a, str) for a in corpus]), "corpus must have strings"
        elif isinstance(corpus, pd.Series):
            assert len(corpus) == sum([isinstance(a, str) for a in corpus.values]), "corpus must have strings"

        tokenizer_options = ", ".join(self.tokenizers.keys())
        assert tokenizer in self.tokenizers, "valid tokenizers: {}".format(tokenizer_options)

        assert isinstance(vocab_size, int), "vocab_size must be an integer"

        assert max_df >= 0. and max_df <= 1, "max_df must be between 0 and 1"

        self.tokenizer = self.tokenizers[tokenizer]

        self.fit(corpus, vocab_size=vocab_size, max_df=max_df)

        self.N = len(corpus)
        self.K = len(self.vocab)

        self.lengths = [len(d) for d in self.X]

        word_freq_pairs = [(self.vocab[id_], f) for id_, f in self.vocab.cfs.items()]
        sorted_vocab = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)
        self.vocab_by_freq, _ = zip(*sorted_vocab)


    def get_tokenized(self, corpus):
        if isinstance(corpus, pd.Series):
            corpus = corpus.values.tolist()
        if isinstance(corpus[0], list):
            return corpus
        tokens = [self.tokenizer(doc) for doc in corpus]
        return tokens


    def fit(self, corpus, store_doc2bow=True, max_df=0.5, vocab_size=10000):
        if isinstance(corpus, pd.Series):
            corpus = corpus.values.tolist()
        tokens = self.get_tokenized(corpus=corpus)
        vocab = DictGensim(tokens)
        vocab.filter_extremes(no_above=max_df)
        vocab.compactify()
        self.vocab = vocab

        if store_doc2bow:
            self.X = [self.vocab.doc2bow(doc) for doc in tokens]

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
            .format(self.N, self.K, min_length, max_length,
                    median_length, mean_length,
                    " ".join([str(a) for a in list(self.vocab_by_freq)[:20]]))


class TFIDF(DocTerm):
    def __init__(self, corpus, **kwargs):
        super().__init__(corpus, **kwargs)
        self.tfidf_model = TfidfModel(self.X, id2word=self.vocab)

        self.X = self.transform(corpus)


    def transform(self, corpus):

        tokenized = self.get_tokenized(corpus)
        bow = [self.vocab.doc2bow(doc) for doc in tokenized]
        docs = [self.tfidf_model.__getitem__(doc) for doc in bow]

        return corpus2csc(docs, num_terms=len(self.vocab))



class LDA(DocTerm):
    def __init__(self, corpus, method='variational', num_topics=50, num_iterations=500,
                 optimize_interval=10, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__(corpus, **kwargs)
        self.method = method
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.optimize_interval = optimize_interval  # hyperparameters

        self.model = self.__fit_lda_model()
    
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

    def __fit_lda_model(self):

        model = None #
        if self.method == 'variational':
            model = None #
        #if self.method == 'gibbs':
            #model = lda.LDA(n_topics=self.num_topics, n_iter=self.lda_max_iter)
        elif self.method == 'mallet':
            if 'mallet_path' not in self.__dict__:
                raise ValueError("Cannot use mallet without setting \'mallet_path\'")
            model = LdaMallet(mallet_path=self.mallet_path,
                              #prefix=prefix,
                              corpus=self.X,
                              id2word=self.vocab,
                              iterations=self.num_iterations,
                              #workers=4,
                              num_topics=self.num_topics,
                              optimize_interval=self.optimize_interval)
            model = malletmodel2ldamodel(model)
        return model

    def transform(self, corpus):
        
        tokens = self.get_tokenized(corpus)
        corpus = [self.vocab.doc2bow(doc) for doc in tokens]
        
