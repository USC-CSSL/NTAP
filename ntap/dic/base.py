from collections import Counter, defaultdict
import os
import re
import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ._utils import DictionaryLoader
from ntap.utils import Preprocessor
from ntap.bagofwords import DocTerm
from ntap.embed import Embedding
from ntap.summarize import _TableMakerMixin
from ntap.validation import _CheckParamsMixin

logger = logging.getLogger(__name__)

class Dictionary(_CheckParamsMixin, _TableMakerMixin):

    """ Supports lexicon-approaches such as word counting

    Using ``liwc`` package, reads liwc-style dictionaries from
    file (ending in ``.dic``). The LIWC dictionaries are 
    proprietary, and can be purchased from <https://liwc.wpengine.com/>.

    Parameters
    ----------
    name : str
        The name of the dictionary, used to either download (based on 
        a preset list of options from NTAP), load from local file, or 
        both download and load.

        Currently supported options:

        * mfd: Maps to the Moral Foundations Dictionary (MFD)
        * More to come!

        If reading dictionary from a (previously downloaded) local file, 
        name is simply used for print-outs and logging. See 
        ``local_dic_path`` in the ``load`` function, below.
    base_dir : str
        Which local directory to download files to, and from which to 
        load dictionary files. Defaults to ``ntap_data`` in the user's
        home directory
    tokenizer : str
        Function descriptor of tokenization to perform when applying
        Dictionary to text. Defaults to basic word-based tokenizer 
        (see ntap.parse.Tokenizer)
    preprocess : str
        Function descriptor of text cleaning, used similarly to 
        tokenization argument. Defaults to removing non-words 
        (links, mentions, etc.) and transforming to lowercase. See
        ntap.parse.Preprocess for more.


    """

    def __init__(self,
                 name,
                 base_dir="~/ntap_data",
                 preprocess='social+lowercase'):

        self.name = name
        self.base_dir = base_dir
        self.docterm = DocTerm(preprocess=preprocess)

        self.load()

    def load(self, local_dic_path=None):
        """ Read from a .dic file in base_dir or from local file

        Read dictionary file from a .dic format. With default behavior, 
        attempts to locate file from ``self.base_dir`` directory, and 
        downloading if file is available.

        Parameters
        ----------
        local_dic_path : str
            If not None, local path of a .dic file
        """

        if local_dic_path is not None and os.path.exists(local_dic_path):
            local_dir = os.path.dirname(os.path.abspath(__file__))
            loader = DictionaryLoader(base_dir=local_dir)
            self.parser, self.categories = loader.load(local_dic_path) 
            self.lexicon = loader.lexicon
            self.source = local_dic_path
            #logger.exception(f'Could not load {local_dic_path}')

        else:

            loader = DictionaryLoader(base_dir=self.base_dir)
            dic_path = loader.download(self.name)
            self.parser, self.categories = loader.load(dic_path)
            self.lexicon = loader.lexicon
            self.source = dic_path

        #self.dic_items = [[ngram.replace(' ', '_') for ngram in l] for l in self.dic_items]

    def __repr__(self):
        from pprint import pformat
        p_vars = {k: v for k, v in vars(self).items() if k not in {'parser', 'base_dir'}}
        return pformat(p_vars, compact=True)

    def _make_main_table_elements(self):
        data, _ = self.docterm._make_main_table_elements()
        uniq_terms = set(term for sublist in self.lexicon.values() for term in sublist)
        covered_vocab = self.docterm._terms_covered_by_vocab(uniq_terms)

        newdata = {
            'Dictionary': [f'{self.name} ({len(self.categories)} '
                           f'categories, {len(uniq_terms)} terms)'],
            'Read from': [self.source],
            'Categories': [', '.join(self.categories[:3]) + '...']
        }

        #for category, words in self.lexicon.items():
            #data[f'{category} ({len(words)})'] = [', '.join(words[:3]) + '...']

        return newdata, None

    def _make_content_table_elements(self):

        prevdata, _ = self.docterm._make_content_table_elements()
        newdata = dict()
        uniq_terms = set(term for sublist in self.lexicon.values() for term in sublist)
        covered_vocab = self.docterm._terms_covered_by_vocab(uniq_terms)

        # note for later: should have one-liner functionality to do
        # traditional liwc analysis (regression table)
        newdata['DocTerm'] = prevdata['DocTerm']
        newdata['Sparsity'] = prevdata['Sparsity']
        newdata['Covered by lexicon'] = [f'{len(covered_vocab)/len(uniq_terms):.2%}']

        return newdata, None

    def fit(self, corpus):

        self.docterm.fit(corpus)
        self.corpus_as_counts = self.transform(corpus)

        return self

    def transform(self, corpus):
        """ Apply dictionary to corpus by counting pattern matches

        Apply the stored dictionary to a new corpus of documents, returning a 
        sparse array in compressed sparse row (csr) format. 

        Parameters
        ----------
        corpus: Union[Iterable[str], pd.Series]
            List-like object with strings. 

        Returns
        -------
        scipy.sparse.csr_matrix
            Resulting dictionary counts in compressed sparse row 
            format. To convert to dense (producing lots of zeros in 
            matrix), you can call ``.todense`` on the object. See
            documentation in scipy.sparse

        """

        dic_docs = list()
        lengths = list()
        for doc in self.docterm.corpus_as_tokens:
            counts = dict(Counter(cat for token in doc for cat in self.parser(token)))
            N = len(doc)
            for cat in self.categories:
                if cat not in counts:
                    counts[cat]= 0
                if N > 0:
                    counts[cat] /= N
            dic_docs.append(counts)
        dic_docs = pd.DataFrame(dic_docs)
        return csr_matrix(dic_docs.values)

def _cosine_matrix(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    return np.einsum('ij,kj->ik', X, Y) / np.einsum('i,j->ij', X_norm, Y_norm)

class _Dic_Centers:

    def __init__(self, embed_name, dic_name):
        self.embed_name = embed_name
        self.dic_name = dic_name

    def load(self, embedding, dic):

        self.dic_vecs = dict()
        self.dic_center_words = dict()

        vocab = {w: idx for idx, w in enumerate(embedding.vocab)}
        for name, dic_items in dic.lexicon.items():
            dic_words = [w for w in dic_items if '*' not in dic_items]
            dic_stems = [w.replace('*', '') for w in dic_items if '*' in w]

            matched_words = [vocab[w] for w in dic_words if w in vocab]

            stem_regex = re.compile(r'\B(?:{})[\w]+\B'.format('|'.join(dic_stems)))
            matched_stems = stem_regex.findall(" ".join(list(vocab.keys())))
            matched_stems = [vocab[w] for w in matched_stems if w in vocab]

            matched_idxs = list(set(matched_words + matched_stems))

            self.dic_vecs[name] = embedding.vecs[matched_idxs, :]
            self.dic_center_words[name] = [embedding.vocab[i] for i in matched_idxs]

    def get_centers(self):

        centers = {name: vecs.mean(axis=0) for name, vecs in self.dic_vecs.items()}
        centers = sorted(centers.items(), key=lambda x: x[0])

        return zip(*centers)

class DDR:
    """ Distributed Dictionary Representations

    Implements Garten et al. (2018). Dictionary words are mapped to
    word embeddings, and subsequently each dictionary category is
    represented by the average of the mapped word embeddings.

    Implements a ``transform`` method that computes the geometric 
    similarity of a document's word embeddings to each dictionary.

    Parameters
    ----------
    dic : str
        Name of a dictionary or path to a dictionary file (.dic)
    embedding : str
        Embedding to use. Corresponds to available embeddings via
        the Embedding class. By default, embedding files are loaded
        from the ``~/ntap_data`` directory. This can be overriden via
        \*\*kwargs.
    preprocessor : str
        How new documents are processed before tokenization. Defaults 
        to full set of ``clean`` operations (see Preprocessor class 
        in ntap.parse)
    tokenizer : str
        How words in new documents are tokenized. Defaults to simple
        word regex (see Tokenizer class in ntap.parse)
    **kwargs
        Optional arbitrary named parameters. Accepts parameters for
        Embedding and Dictionary constructors


    """

    def __init__(self,
                 dic,
                 embedding='glove-wiki',
                 preprocessor='clean+lowercase',
                 tokenizer='word',
                 **kwargs):

        self.embed = Embedding(name=embedding, **kwargs)
        self.dic = Dictionary(name=dic, **kwargs)
        self.tokenizer = Tokenizer(tokenizer)
        self.preprocessor = Preprocessor(preprocessor)

        self.embedded_dic = _Dic_Centers(embedding, dic)
        self.embedded_dic.load(self.embed, self.dic)

        self.names, self.centers = self.embedded_dic.get_centers()

    def transform(self, data, **kwargs):
        """ Compute similarities between docs and dictionaries

        Apply stored dictionary and word embeddings to new documents, 
        generating cosine similarities between dictionary centers and
        document centers for each document.

        """
        tokenized = self.preprocessor.transform(data)
        embed_docs = self.embed.transform(tokenized)

        return _cosine_matrix(embed_docs, self.centers)

