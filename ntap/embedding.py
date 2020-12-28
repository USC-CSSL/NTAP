import re
import warnings

import numpy as np
import pandas as pd

from ntap.bagofwords import Dictionary, DocTerm
from ntap.utils import open_embed

class Embedding:

    def __init__(self, name='glove-wiki', **kwargs):

        self.__dict__.update(kwargs)

        self.embed_name = name
        self.load_from_ntap_dir(name)
        #self.X = self.fit_transform(corpus)

    def load_from_ntap_dir(self, name):

        if 'vec_size' in self.__dict__:
            self.vocab, self.vecs = open_embed(name, vec_size=self.vec_size)
        else:
            self.vocab, self.vecs = open_embed(name)

    def transform(self, corpus, min_words=0, convert_to_nan=True):

        if not isinstance(corpus, DocTerm):

            #_verify_corpus(corpus)
            docterm_params = {k: v for k, v in self.__dict__.items() 
                              if k in {'vocab_size', 'max_df'}}
            dt = DocTerm(**docterm_params)
        else:
            dt = corpus

        if not hasattr(dt, 'bow'):  # not fit object
            dt.fit(corpus)

        self.empty_indices = list()
        self.found_words = set()
        self.missing_words = set()

        embedded_docs = np.zeros((len(dt), self.vectors.vector_size))

        for i, doc_tokenized in enumerate(dt.tokenized):
            c = 0
            for t in doc_tokenized:
                if t in self.vectors:
                    embedded_docs[i, :] += self.vectors[t]
                    c += 1
                    self.found_words.add(t)
                else:
                    self.missing_words.add(t)
            if c == 0:
                warnings.warn('Empty doc found while computing embedding', RuntimeWarning)
                self.empty_indices.append(i)
            else:
                embedded_docs[i, :] /= c

        if convert_to_nan:
            embedded_docs[self.empty_indices, :] = np.NaN

        return embedded_docs

    """
    def __str__(self):
        top_missing = sorted(self.missing_words)
        return ("Embedding Object (documents: {})\n"
                "Embedding File Used: {}\n"
                "Unique Embeddings found: {}\n"
                "Missing Words ({}): {}...\n"
                "{} Docs had no embeddings").format(len(self),
                                                   self.embed_name,
                                                   len(self.found_words),
                                                   len(top_missing),
                                                   " ".join(top_missing[:10]),
                                                   len(self.empty_indices))
    """

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

    def __init__(self, dic, **kwargs):

        self.__dict__.update(kwargs)

        self.emb_corpus = Embedding(**kwargs)

        if not isinstance(dic, Dictionary) and isinstance(dic, str):
            self.dic = Dictionary(dic_path=dic)
        elif isinstance(dic, Dictionary):
            self.dic = dic

        self.emb_dic = _Dic_Centers(self.emb_corpus.vectors, self.dic)

        #self.X = self.fit()

    def fit(self, **kwargs):
        #doc_avgs[~np.isnan(doc_avgs).any(axis=1)]
        return _cosine_matrix(self.emb_corpus.X, self.emb_dic.centers)
