
### Helper function ###
def load_glove_from_file(fname, vocab=None):
    # vocab is possibly a set of words in the raw text corpus
    if not os.path.isfile(fname):
        raise IOError("You're trying to access a GloVe embeddings file that doesn't exist")
    embeddings = dict()
    with open(fname, 'r') as fo:
        for line in fo:
            tokens = line.split()
            if vocab is not None:
                if tokens[0] not in vocab:
                    continue
            if len(tokens) > 0:
                embeddings[str(tokens[0])] = np.array(tokens[1:], dtype=np.float32)
    return embeddings

import os, json
from gensim.models import KeyedVectors as kv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

glove_path = os.environ["GLOVE_PATH"]
word2vec_path = os.environ["WORD2VEC_PATH"]
dictionary_path = os.environ["DICTIONARIES"]

class DDRVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_type, stop_words,
                 tokenizer, dictionary, similarity):
        self.embedding_type = embedding_type
        self.stoplist = set(stop_words) if stop_words is not None else None
        if self.stoplist is None:
            print("WHY THE FUCK IS IT NONE")
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        if embedding_type == 'glove':
            self.embeddings_path = glove_path
        else:
            self.embeddings_path = word2vec_path
        dict_path = os.path.join(dictionary_path, dictionary + '.json')
        try:
            with open(dict_path, 'r') as fo:
                data = json.load(fo)
                self.dictionary = data.items()
        except FileNotFoundError:
            print("Could not load dictionary %s from %s" % (self.dictionary, dict_path))
            exit(1)
        self.similarity = similarity

    def get_feature_names(self):
        return [item[0] for item in self.dictionary]

    def get_document_avg(self, tokens, embed_size=300, min_threshold=0):
        arrays = list()
        oov = list()
        count = 0
        for token in tokens:
            if self.embedding_type == 'word2vec':
                try:
                    array = self.skipgram_vectors.get_vector(token)
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            elif self.embedding_type == 'glove':
                try:
                    array = self.glove_vectors[token]
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            else:
                raise ValueError("Incorrect embedding_type specified; only possibilities are 'skipgram and 'GloVe'")
        if count <= min_threshold:
            return np.random.rand(embed_size), oov
        sentence = np.array(arrays)
        mean = np.mean(sentence, axis=0)
        return mean, oov

    def fit(self, X, y=None):
        """
        Load selected word embeddings based on specified name 
            (raise exception if not found)
        """

        if not os.path.exists(self.embeddings_path):
            print(self.embeddings_path)
            raise ValueError("Invalid Word2Vec training corpus + method")
        print("Loading embeddings")
        if self.embedding_type == 'word2vec':
            # type is 'gensim.models.keyedvectors.Word2VecKeyedVectors'
            self.skipgram_vectors = kv.load_word2vec_format(self.embeddings_path, binary=True)
        if self.embedding_type == 'glove':
            # type is dict
            self.glove_vectors = load_glove_from_file(self.embeddings_path)
        return self

    def transform(self, raw_docs, y=None):
        print("Calculating dictionary centers")
        concepts = list()
        for concept, words in self.dictionary:
            concept_mean, _ = self.get_document_avg(words)
            concepts.append(concept_mean)
        ddr_vectors = list()
        for sentence in raw_docs:
            if self.stoplist is not None:
                tokens = list(set(self.tokenizer(sentence)) - self.stoplist)
            sentence_mean, oov = self.get_document_avg(tokens)    
            outputs = [self.similarity(sentence_mean, concept) for concept in concepts]
            ddr_vectors.append(outputs)
        X = np.array(ddr_vectors)
        X = np.nan_to_num(X)
        return X
