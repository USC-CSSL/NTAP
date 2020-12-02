import pandas as pd
import abc

import numpy as np
import json, re, os, tempfile, sys, io, gzip
#from nltk import tokenize as nltk_token
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer 
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


#stem = SnowballStemmer("english").stem

#tokenizers = {'treebank': nltk_token.TreebankWordTokenizer().tokenize,
              #'wordpunct': nltk_token.WordPunctTokenizer().tokenize,
              #'tweettokenize': nltk_token.TweetTokenizer().tokenize}


def _read_liwc_dictionary(liwc_file):
    cates = {}
    words = {}
    percent_count = 0

    for line in liwc_file:
        line_stp = line.strip()
        if line_stp:
            parts = line_stp.split('\t')
            if parts[0] == '%':
                percent_count += 1
            else:
                if percent_count == 1:
                    cates[parts[0]] = parts[1]
                    words[parts[0]] = []
                else:
                    for cat_id in parts[1:]:
                        words[cat_id].append(parts[0])
    items = []
    categories = []
    for cat_id in cates:
        categories.append(cates[cat_id])
        items.append(words[cat_id])
    return tuple(categories), tuple(items)

def open_dictionary(dictionary_path):
    if not os.path.exists(dictionary_path):
        raise ValueError("Dictionary not found at {}".format(dictionary_path))
    if dictionary_path.endswith(".json"):
        try:
            with open(dictionary_path, 'r') as fo:
                dictionary = json.load(fo)  # {category: words}
                categories, items = zip(*sorted(dictionary.items(), key=lambda x:x[0]))
                return categories, items
        except Exception:
            raise ValueError("Could not import json dictionary")
    elif dictionary_path.endswith(".dic"):  # traditional LIWC format
        
        try:
            with open(dictionary_path, 'r') as liwc_file:
                categories, items = _read_liwc_dictionary(liwc_file)
                return categories, items

        except Exception:
            raise ValueError("Cound not import liwc dictionary")
    else:
        raise ValueError("Dictionary type not supported")
    

# Classes: Preprocessor, DocTermMatrix

# idea: chain preprocessing functions together. Strings or literal functions

# language: 'en'
# remove_rules (list of functions or strings)
# defaults: [all_punc, stopwords, numbers]
"""

    def __encode(self, column):
        self.__truncate_count = 0
        self.__pad_count = 0
        self.__unk_count = 0
        self.__token_count = 0
        tokenized = [None for _ in range(len(self.data))]

        self.sequence_lengths = list()

        for i, (_, string) in enumerate(self.data[column].iteritems()):
            tokens = self.__tokenize_doc(string)
            self.sequence_lengths.append(len(tokens))
            tokenized[i] = self.__encode_doc(tokens)

        #self.max_len = max(self.sequence_lengths)

        #print("Encoded {} docs".format(len(tokenized)))
        #print("{} tokens lost to truncation".format(self.__truncate_count))
        #print("{} padding tokens added".format(self.__pad_count))
        #print("{:.3%} tokens covered by vocabulary of size {}".format(
            #(self.__token_count - self.__unk_count) / self.__token_count, len(self.vocab)))
        self.sequence_data = np.array(tokenized)
        self.num_sequences = len(tokenized)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)


    def __clean_doc_fn(self, OP):
        self.__truncate_count += max(len(doc) - self.max_len, 0)
        unk_idx = self.mapping["<UNK>"]
        pad_idx = self.mapping["<PAD>"]
        encoded = [pad_idx] * len(doc) #self.max_len
        self.__pad_count += max(0, self.max_len - len(doc))
        for i in range(min(self.max_len, len(doc))):  # tokenized
            encoded[i] = self.mapping[doc[i]] if doc[i] in self.mapping else unk_idx
            self.__unk_count += int(encoded[i] == unk_idx)
            self.__token_count += int((encoded[i] != pad_idx) & (encoded[i] != unk_idx))
        return np.array(encoded, dtype=np.int32)

    def encode_targets(self, columns, normalize=None):

        if not isinstance(columns, list):
            columns = [columns]
        for c in columns:
            if c not in self.data.columns:
                raise ValueError("Column not in Data: {}".format(c))
            if encoding == 'one-hot':
                enc = OneHotEncoder(sparse=False, categories='auto')
                X = [ [v] for v in self.data[c].values]
                X_onehot = enc.fit_transform(X)
                target_names = enc.get_feature_names().tolist()
                target_names = [f.split('_')[-1] for f in target_names]
                self.target_names[c] = target_names
                self.targets[c] = X_onehot
                #self.weights[c] = {name: sum(self.targets[c] == name) for \
                        #name in self.target_names[c]}
            else:
                enc = LabelEncoder()
                X = self.data[c].values.tolist()
                X_enc = enc.fit_transform(X)
                self.target_names[c] = enc.classes_
                self.targets[c] = X_enc
                length = len(self.targets[c])
                self.weights[c] = [(length - sum(self.targets[c] == name))/length for name in self.target_names[c]]

    def encode_inputs(self, columns, var_type='categorical', normalize=None, encoding='one-hot'):

        if not isinstance(columns, list):
            columns = [columns]
        for c in columns:
            if c not in self.data.columns:
                raise ValueError("Column not in Data: {}".format(c))
            if encoding == 'one-hot':
                enc = OneHotEncoder(sparse=False, categories='auto')
                X = [ [v] for v in self.data[c].values]
                X_onehot = enc.fit_transform(X)
                feat_names = enc.get_feature_names().tolist()
                feat_names = [f.split('_')[-1] for f in feat_names]
                self.feature_names[c] = feat_names
                self.features[c] = X_onehot
            else:
                enc = LabelEncoder()
                X = self.data[c].values.tolist()
                X_enc = enc.fit_transform(X)
                self.feature_names[c] = enc.classes_
                self.features[c] = X_enc
"""

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

        vocab = Dictionary(self.tokens)
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
        

class DDR(DocTerm):
    def __init__(self, corpus, dic, glove_path, **kwargs):
        dictionary = Dictionary(dic)

        for word_list in name:
            for w in word_list:
                if w not in self.mapping:
                    self.mapping[w] = len(self.mapping)
                    self.vocab.append(w)

        embeddings, _ = self.__read_glove(self.glove_path)

        dictionary_centers = {cat: self.__embedding_of_doc(words, embeddings) \
                for cat, words in zip(dictionary, name)}
        features = list()
        for doc in self.data[column].values.tolist():
            e = self.__embedding_of_doc(doc, embeddings=embeddings)
            not_found = False
            if e.sum() == 0:  # no tokens found
                not_found = True
            doc_feat = dict()
            for category, dict_vec in dictionary_centers.items():
                if not_found:
                    e = np.random.rand(len(e))
                doc_feat[category] = cosine(dict_vec, e)
            features.append(doc_feat)

        features = pd.DataFrame(features)
        features, categories = features.values, list(features.columns)
        self.features["ddr"] = np.array(features)  # type numpy array
        self.feature_names["ddr"] = categories # list of strings

    def fit(self, X, y=None):
        pass
    def transform(self, X):
        pass
    """
    def __embedding_of_doc(self, doc_string, embeddings, agg='mean', thresh=1):
        tokens = self.__tokenize_doc(doc_string)
        embedded = list()
        for t in tokens:
            if t in self.mapping:
                embedded.append(embeddings[self.mapping[t]])
        if len(embedded) < thresh:
            return np.zeros(embeddings.shape[1])
        return np.array(embedded).mean(axis=0)
    """
