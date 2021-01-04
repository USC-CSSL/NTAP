import re
import os
import logging
import warnings

import numpy as np
import pandas as pd

from ntap.bagofwords import DocTerm
from ._utils import EmbeddingLoader

log = logging.getLogger(__name__)

class Embedding:
    """ Background class for managing static word embeddings

    Parameters
    ----------
    name : str
        One of "{glove|fasttext}-{wiki|cc}", e.g., glove-wiki
    vec_size : int
        Only defined for name=glove-wiki, else ignored.
        Options: 50, 100, 200, 300
    base_dir : str
        Will attempt to download and load full embedding files 
        from this directory. The default base_dir (~/ntap_data),
        which routes to the user's *home* directory, can be 
        overridden by passing a different directory as `base_dir`,
        or alternatively as an environment variable ``NTAP_BASE_DIR``.
    local_dir : str
        Embedding files will be saved and optionally loaded from 
        this directory. 

    Methods
    -------
    load:
        Reads embedding data from file, executing download if not found.
    save:
        Save to file. Use after ``update_vocab`` to save a smaller, local
        version of embedding for a given project.
    update_vocab:
        Given a corpus, shrink vocab (and embedding matrix). This saves 
        memory and makes lookups more efficient.
    transform:
        Given a set of documents, encode them to averaged embeddings

    """

    def __init__(self, name='glove-wiki', vec_size=300, 
                 base_dir="~/ntap_data", local_dir="./output"):

        self.base_dir = base_dir
        self.local_dir = local_dir
        self.embed_name = name
        if self.embed_name == 'glove-wiki':
            self.vec_size = vec_size  
        else:
            self.vec_size = 300
        self.load()

    def __str__(self):
        return (f"{self.embed_name} vectors loaded from {self.base_dir}\n"
                f"Embedding dimensions: {len(self.vocab)} X {self.vec_size}")

    def load(self, prefer_local=True):
        """ Attempt to load from local_dir then from base_dir

        If no ``vocab.txt`` and ``vecs.npy`` files are found in
        ``self.local_dir/self.embed_name/self.vec_size/``, will 
        attempt to instead load from ``self.base_dir``. If ``vocab.txt``
        and ``vecs.npy`` are not found there, will download and 
        convert files

        Parameters
        ----------
        prefer_local : bool
            Whether or not to check local project directory (set 
            in ``__init__``) before loading full embedding files 
            from base directory (~/ntap_data)
        """

        if prefer_local:
            loader = EmbeddingLoader(self.local_dir)
            if loader.is_converted(self.embed_name, self.vec_size):
                self.vocab, self.vecs = loader.load(self.embed_name, 
                                                        self.vec_size)
                return

        loader = EmbeddingLoader(self.base_dir)
        loc = loader.download(name_of_embedding=self.embed_name)
        if not loader.is_converted(self.embed_name, self.vec_size):
            loader.extract(self.embed_name, loc, vec_size=self.vec_size)
        self.vocab, self.vecs = loader.load(self.embed_name, self.vec_size)

    def save(self):
        """ Save vocab and vector file (.npy) to self.local_dir """

        embed_dir = os.path.join(self.local_dir, self.embed_name, 
                                 str(self.vec_size))
        if not os.path.isdir(embed_dir):
            log.info(f"Creating {embed_dir}")
            os.makedirs(embed_dir)
        self.vocab_path = os.path.join(embed_dir, "vocab.txt")
        self.vec_path = os.path.join(embed_dir, "vecs.npy")

        log.info(f"Saving vocab (n={len(self.vocab)}) to {self.vocab_path}")
        with open(self.vocab_path, 'w') as fo:
            log.info(f"Writing vocab file at {self.vocab_path}")
            fo.write('\n'.join(str(w) for w in self.vocab))

        n, k = self.vecs.shape
        log.info(f"Saving {n} X {k} embeddings to {self.vec_path}")
        np.save(self.vec_path, self.vecs)

    def update_vocab(self, corpus, vocab_size=10000, 
                     max_df=0.5, save_after=True):
        """ Shrink vocab to corpus and prune unused embeddings

        Parameters
        ----------
        corpus : :obj:`list` of :obj:`str`
            An iterable of documents in string form
        vocab_size : :obj:`int`
            The most frequent ``vocab_size`` from corpus are kept.
        max_df : :obj:`float`
            Words that occur in more than max_df (ratio 0--1) are pruned
        save_after : bool
            If True, will automatically save updated vocab and vectors 
            to ``self.local_dir``
        """

        dt = DocTerm(vocab_size=vocab_size, max_df=max_df)
        dt.fit(corpus)
        fitted_vocab = set(dt.top_vocab(k=len(dt.vocab)))
        prev_len = len(self.vocab)
        prev_vocab_map = {w: i for i, w in enumerate(self.vocab)}

        self.vocab = [w for w in self.vocab if w in fitted_vocab]
        updated_indices = [prev_vocab_map[w] for w in self.vocab 
                           if w in prev_vocab_map]
        self.vecs = self.vecs[updated_indices, :]

        if prev_len != len(self.vocab) and save_after:
            save_path = os.path.join(self.local_dir, self.embed_name, 
                                     str(self.vec_size), "vocab.txt")
            log.info(f"Updating vocab at {save_path}, "
                     f"{prev_len}->{len(self.vocab)}")


    def transform(self, docs, min_words=0, is_tokenized=True):
        """ Map a list of docs to their avg word embedding

        Primarily for use by methods like DDR and as a feature baseline
        for supervised models.

        Parameters
        ----------
        docs : list-like
            List of strings (if not is_tokenized) or lists
        min_words : int
            Docs with fewer than min_words tokens in embedding
            (self.vocab) are replaced with NaN
        is_tokenized :
            If True, docs is expected to contain token-lists rather
            than str

        Returns
        -------
        numpy.ndarray
            A matrix size N x self.vec_size, where N is length of docs

            Missing docs (see ``min_words``) consistent of a row of NaN

        """

        if is_tokenized:
            tokenized_docs = docs
        else:
            cleaned = self.preprocessor.transform(docs)
            tokenized_docs = self.tokenizer.transform(cleaned)

        lookup = {w: idx for idx, w in enumerate(self.vocab)}

        empty_indices = list()
        missing = set()

        embedded_docs = np.zeros((len(tokenized_docs), self.vec_size))

        for i, tokens in enumerate(tokenized_docs):

            doc_ids = {lookup[w] for w in tokens if w in lookup}
            missing.update({w for w in tokens if w not in lookup})

            if len(doc_ids) <= min_words:
                #warnings.warn('Empty doc found while computing embedding', RuntimeWarning)
                embedded_docs[i, :] = np.NaN
            else:
                doc_vecs = self.vecs[list(doc_ids), :]
                embedded_docs[i, :] = doc_vecs.mean(axis=0)

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


