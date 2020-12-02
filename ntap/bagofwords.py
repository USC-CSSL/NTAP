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
from scipy.spatial.distance import cosine

# local imports
from .utils import read_dictionary

class Dictionary:

    def __init__(self, dic_path=None, dic_data=None):

        if dic_path is not None:
            with open(dic_path) as liwc_file:
                self.categories, self.items = read_dictionary(liwc_file)
        elif dic_data is not None:
            self.categories, self.items = dic_data
        else:
            self.dic = None

        if self.dic is not None:
            self.dic = {c: items for c, items in zip(self.categories, self.items)}
            self.__build_regexes()

    def __build_regexes(self):
        """ Initialize regexes for word searching """
        return

# chain preprocessing functions 

# language: 'en'
# remove_rules (list of functions or strings)
# defaults: [all_punc, stopwords, numbers]

word_regex = re.compile(r'[\w\_]{2,15}')
class DocTerm:

    tokenizers = {'regex': lambda x: word_regex.findall(x)}

    def __init__(self, corpus, tokenizer='regex', vocab_size=10000, 
                 max_df=0.5, lang='english', **kwargs):
        """ corpus is list-like; contains str documents """
        if isinstance(corpus, pd.Series):
            corpus = corpus.values.tolist()
        self.docs = corpus
        self.N = len(corpus)
        self.tokens = [self.tokenizers[tokenizer](doc) for doc in corpus]

        lengths = [len(d) for d in self.tokens]
        self.min_length = min(lengths)
        self.max_length = max(lengths)
        self.median_length = np.median(lengths)
        self.mean_length = np.mean(lengths)

        vocab = DictGensim(self.tokens)
        vocab.filter_extremes(no_above=max_df)
        vocab.compactify()
        self.X = [vocab.doc2bow(doc) for doc in self.tokens]

        self.K = len(vocab)
        self.vocab = vocab
        word_freq_pairs = [(self.vocab[id_], f) for id_, f in self.vocab.cfs.items()]
        sorted_vocab = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)
        self.vocab_by_freq, _ = zip(*sorted_vocab)

    def __str__(self):
        return ("DocTerm Object (documents: {}, terms: {})\n"
                "Doc Lengths ({}-{}): mean {:.2f}, median {:.2f}\n"
                "Top Terms: {}") \
            .format(self.N, self.K, self.min_length, self.max_length,
                    self.median_length, self.mean_length,
                    " ".join([str(a) for a in list(self.vocab_by_freq)[:20]]))

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
        
        tokens = [self.tokenizers[self.tokenizer](doc) for doc in corpus]
        corpus = [self.vocab.doc2bow(doc) for doc in tokens]
        
