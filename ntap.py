"""
file: ntap.py
about: contains methods and classes available from base ntap directory
    - class Dataset
    - tokenization methods
"""

MALLET_PATH = "/home/brendan/mallet-2.0.8/bin/mallet"


import pandas as pd
import numpy as np
import json, re, os, tempfile
from nltk import tokenize as nltk_token
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models.wrappers import LdaMallet

stem = SnowballStemmer("english").stem

link_re = re.compile(r"(http(s)?[^\s]*)|(pic\.[s]*)")
hashtag_re = re.compile(r"#[a-zA-Z0-9_]+")
mention_re = re.compile(r"@[a-zA-Z0-9_]+")

pat_type = {'links': link_re, 
            'hashtags': hashtag_re,
            'mentions': mention_re}

tokenizers = {'treebank': nltk_token.TreebankWordTokenizer().tokenize,
              'wordpunct': nltk_token.WordPunctTokenizer().tokenize,
              'tweettokenize': nltk_token.TweetTokenizer().tokenize}

#from tokenization.happierfuntokenizing import HappierTokenizer
#def happiertokenize(text):
    #tok = HappierTokenizer(preserve_case=False)
    #return tok.tokenize(text)

def read_file(path):
    if not os.path.exists(path):
        raise ValueError("Path does not point to existing file: {}".format(path))
        return
    ending = path.split('.')[-1]
    if ending == 'csv':
        return pd.read_csv(path)
    elif ending == 'tsv':
        return pd.read_csv(path, delimiter='\t')
    elif ending == 'pkl':
        return pd.read_pickle(path)
    elif ending == 'json':
        return pd.read_json(path)


class Dataset:
    def __init__(self, path, tokenizer='wordpunct', vocab_size=5000,
            embed='glove', min_token=5, stopwords=None, stem=False,
            lower=True, max_len=100):
        try:
            self.data = read_file(path)
        except Exception as e:
            print("Exception:", e)
            return
        print("Loaded file with {} documents".format(len(self.data)))
        self.min_token = min_token
        self.embed_source = embed
        self.vocab_size = vocab_size
        self.tokenizer = tokenizers[tokenizer]
        self.lower = lower
        if isinstance(stopwords, list) or isinstance(stopwords, set):
            self.stopwords = set(stopwords)
        elif stopwords == 'nltk':
            self.stopwords = set(stopwords.words('english'))
        elif stopwords is None:
            self.stopwords = set()
        else:
            raise ValueError("Unsupported stopword list: {}\nOptions include: nltk".format(stopwords))
        self.stem = stem
        self.max_len = max_len

    """
    method encode_docs: given column, tokenize and save documents as list of
    word IDs in self.docs
    """
    def encode_docs(self, column, level='word'):
        if column not in self.data.columns:
            raise ValueError("Given column is not in data: {}".format(column))
        # TODO: catch exception where column is numeric (non-text) type

        self.__learn_vocab(column)

        self.__truncate_count = 0
        self.__pad_count = 0
        self.__unk_count = 0
        self.__token_count = 0
        tokenized = [None for _ in range(len(self.data))]
        for i, (_, string) in enumerate(self.data[column].iteritems()):
            tokens = self.__tokenize_doc(string)
            tokenized[i] = self.__encode_doc(tokens)
        print("Encoded {} docs".format(len(tokenized)))
        print("{} tokens lost to truncation".format(self.__truncate_count))
        print("{} padding tokens added".format(self.__pad_count))
        print("{:.3%} tokens covered by vocabulary of size {}".format((self.__token_count -
            self.__unk_count) / self.__token_count, len(self.vocab)))
        self.docs = np.array(tokenized)

    def clean(self, column, remove=["hashtags", "mentions", "links"], mode='remove'):
        if column not in self.data:
            raise ValueError("{} not in dataframe".format(column))
        def mentions(t):
            return mention_re.sub("", t)
        def links(t):
            return link_re.sub("", t)
        def hashtags(t):
            return hashtag_re.sub("", t)

        for pattern in pat_type:
            if pattern == "mentions":
                self.data[column] = self.data[column].apply(mentions)
            if pattern == "hashtags":
                self.data[column] = self.data[column].apply(hashtags)
            if pattern == "links":
                self.data[column] = self.data[column].apply(links)
        prev = len(self.data)
        self.data = self.data[self.data[column].apply(self.__good_doc)]
        print("Removed {} docs after cleaning that didn't have enough valid tokens".format(prev - len(self.data)))

    def set_params(self, **kwargs):
        if "tokenizer" in kwargs:
            self.tokenizer = tokenizers[kwargs["tokenizer"]]
        if "vocab_size" in kwargs:
            self.vocab_size = kwargs["vocab_size"]
        if "stopwords" in kwargs:
            self.stopwords = kwargs["stopwords"]
        if "lower" in kwargs:
            self.lower = kwargs["lower"]
        if "stem" in kwargs:
            self.stem = stem
        if "max_len" in kwargs:
            self.max_len = kwargs["max_len"]

    def __learn_vocab(self, column):
        vocab = dict()
        for doc in self.data[column].values:
            for word in self.__tokenize_doc(doc):
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        top = list(sorted(vocab.items(), key=lambda x: x[1],
            reverse=True))[:self.vocab_size]
        types, counts = zip(*top)
        types = list(types)
        types.append("<PAD>")
        types.append("<UNK>")
        self.vocab = types
        self.__mapping = {word: idx for idx, word in enumerate(self.vocab)}
        
    def __good_doc(self, doc):
        if len(self.tokenizer(doc)) < self.min_token:
            return False
        return True

    def __tokenize_doc(self, doc):
        tokens = self.tokenizer(doc)
        if self.stem:
            filtered_tokens = [stem(w) for w in tokens if w not in
                    self.stopwords]
            if self.lower():
                return [w.lower() for w in filtered_tokens]
            else:
                return filtered_tokens
        else:
            filtered_tokens = [w for w in tokens if w not in self.stopwords]
            if self.lower:
                return [w.lower() for w in filtered_tokens]
            else:
                return filtered_tokens

    def __encode_doc(self, doc):
        self.__truncate_count += max(len(doc) - self.max_len, 0)
        unk_idx = self.__mapping["<UNK>"]
        pad_idx = self.__mapping["<PAD>"]
        encoded = [pad_idx] * self.max_len
        self.__pad_count += max(0, self.max_len - len(doc))
        for i in range(min(self.max_len, len(doc))):  # tokenized
            encoded[i] = self.__mapping[doc[i]] if doc[i] in self.__mapping else unk_idx
            self.__unk_count += int(encoded[i] == unk_idx)
            self.__token_count += int((encoded[i] != pad_idx) & (encoded[i] != unk_idx))
        return np.array(encoded, dtype=np.int32)


    def encode_targets(self, columns, var_type='categorical', normalize=None,
            encode=True):

        self.targets = dict()
        self.__target_mappings = dict()
        if not isinstance(columns, list):
            columns = [columns]

        for c in columns:
            if c not in self.data.columns:
                raise ValueError("Column not in Data: {}".format(c))
            if encode:  
                vals = [str(val) for val in list(set(self.data[c]))]
                self.__target_mappings[c] = {v: i for i, v in enumerate(vals)}
            else:  
                vals = [int(val) for val in list(set(self.data[c]))]
                self.__target_mappings[c] = {str(v): v for v in vals}  # identity
            self.targets[c] = np.array([self.__target_mappings[c][str(val)] for val
                in self.data[c].values])

    def lda(self, stopwords=None, method='mallet', num_topics=20, max_iter=500, save_model=None, load_model=None):
        if not hasattr(self, 'features'):
            self.features = dict()
        if method == 'mallet':
            print("Mallet LDA")
        else:
            raise ValueError("Invalid paramater for LDA.method: {}".format(method))
        tmp_dir = os.path.join(tempfile.gettempdir(), "mallet_lda/")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        replace = set([self.__mapping["<UNK>"], self.__mapping["<PAD>"]])
        docs = self.docs.tolist()
        docs = [[(t, doc.count(t)) for t in list(set(doc) - replace)] for doc in docs]
        print(docs)
        id2word = {v:k for k,v in self.__mapping.items()}
        model = LdaMallet(mallet_path=MALLET_PATH,
                          id2word=id2word,
                          prefix=tmp_dir,
                          num_topics=num_topics,
                          iterations=max_iter,
                          optimize_interval=20)
        model.train(docs)
        doc_topics = list()
        for doc_vec in model.read_doctopics(model.fdoctopics()):
            topic_ids, vecs = zip(*doc_vec)
            doc_topics.append(np.array(vecs))
        self.features["lda"] = np.array(doc_topics)
        self.lda = model.get_topics()
        return

    def tfidf(self, column, stopwords=None, vocab_size=None, **kwargs):
        print("TODO: implement tfidf, save to self.tfidf")

    def ddr(self, column, dictionary, embed='glove', **kwargs):
        print("TODO: implement ddr features")

    def bert(self, some_params):
        print("TODO: implement a featurization based on BERT")


    def write(self, path):

        formatting = path.split('.')[-1]
        if formatting == '.json':
            self.data.to_json(path)
        if formatting == '.csv':
            self.data.to_csv(path)
        if formatting == '.tsv':
            self.data.to_csv(path, sep='\t')
        if formatting == '.pkl':
            self.data.to_pickle(path)
        if formatting == '.stata':
            self.data.to_stata(path)
        if formatting == '.hdf5':
            self.data.to_hdf(path)
        if formatting == '.excel':
            self.data.to_excel(path)
        if formatting == '.sql':
            self.data.to_sql(path)
