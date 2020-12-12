import re
import warnings

import numpy as np
import pandas as pd
import gensim.downloader as api

from ntap.bagofwords import Dictionary, DocTerm
from ntap.utils import load_glove, load_fasttext

class EmbedCorpus:

    model_list = ['fasttext-wiki-news-subwords-300',
                  'glove-twitter-100',
                  'glove-twitter-200',
                  'glove-twitter-25',
                  'glove-twitter-50',
                  'glove-wiki-gigaword-100',
                  'glove-wiki-gigaword-200',
                  'glove-wiki-gigaword-300',
                  'glove-wiki-gigaword-50',
                  'word2vec-google-news-300']


    def __init__(self, corpus, embedding_name='glove-wiki-gigaword-300', 
                 is_tokenized=False, **kwargs):
        if not is_tokenized:
            self.dtm = DocTerm(corpus, **kwargs)
        self.embed_name = embedding_name
        self.is_tokenized = is_tokenized
        if embedding_name in self.model_list:
            self.E = api.load(embedding_name)
        else:
            raise ValueError("{} embedding not recognized by gensim API",
                             "Options: {}".format(embedding_name,
                                                  " ".join(self.model_list)))
        self.X = self.fit_transform(corpus)

    def fit_transform(self, data, min_words=0):
        if isinstance(data, pd.Series):
            data = data.values.tolist()

        self.empty_indices = list()
        self.found_words = set()
        self.missing_words = set()
        if self.is_tokenized:
            tokens = data
        else:
            tokens = self.dtm.get_tokenized(data)  # list of token-lists
        embedded_docs = np.zeros((len(self.dtm), self.E.vector_size))

        for i, doc_tokenized in enumerate(tokens):
            c = 0
            for t in doc_tokenized:
                if t in self.E:
                    embedded_docs[i, :] += self.E[t]
                    c += 1
                    self.found_words.add(t)
                else:
                    self.missing_words.add(t)
            if c == 0:
                warnings.warn('Empty doc in embedding corpus', RuntimeWarning)
                self.empty_indices.append(i)
            else:
                embedded_docs[i, :] /= c
        return embedded_docs

    def __len__(self):
        return len(self.dtm)

    def __str__(self):
        top_missing = sorted(self.missing_words)
        return ("EmbedCorpus Object (documents: {})\n"
                "Embedding File Used: {}\n"
                "Unique Embeddings found: {}\n"
                "Missing Words ({}): {}...\n"
                "{} Docs had no embeddings").format(len(self),
                                                   self.embed_name,
                                                   len(self.found_words),
                                                   len(top_missing),
                                                   " ".join(top_missing[:10]),
                                                   len(self.empty_indices))

def _cosine_matrix(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    return np.einsum('ij,kj->ik', X, Y) / np.einsum('i,j->ij', X_norm, Y_norm)

class _Dic_Centers:
    def __init__(self, embedding, dic_obj):
        self.dic_vecs = dict()
        for name, dic_items in zip(dic_obj.categories, dic_obj.dic_items):
            self.dic_vecs[name] = list()
            print(dic_items)
            for t in dic_items:
                if t in embedding:
                    print(name, t)
                    self.dic_vecs[name].append(embedding[t])
            self.dic_vecs[name] = np.array(self.dic_vecs[name])
        centers = {name: vecs.mean(axis=0) for name, vecs in self.dic_vecs.items()}
        centers = sorted(centers.items(), key=lambda x: x[0])
        self.names, self.centers = zip(*centers)
        self.centers = np.array(self.centers)

class DDR:

    def __init__(self, corpus, dic, **kwargs):

        self.emb_corpus = EmbedCorpus(corpus, **kwargs)

        if not isinstance(dic, Dictionary) and isinstance(dic, str):
            self.dic = Dictionary(dic_path=dic)
        elif isinstance(dic, Dictionary):
            self.dic = dic

        self.emb_dic = _Dic_Centers(self.emb_corpus.E, self.dic)

        self.X = self.fit()

    def fit(self, **kwargs):
        #doc_avgs[~np.isnan(doc_avgs).any(axis=1)]
        return _cosine_matrix(self.emb_corpus.X, self.emb_dic.centers)
