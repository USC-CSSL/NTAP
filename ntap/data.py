"""
file: ntap.py
about: contains methods and classes available from base ntap directory
    - class Dataset
    - tokenization methods
"""

import pandas as pd
import numpy as np
import json, re, os, tempfile, sys, io, gzip
from nltk import tokenize as nltk_token
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer  # TODO: more stemming options
from gensim.models.wrappers import LdaMallet
from gensim.models import TfidfModel
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import copy, inspect
from scipy.spatial.distance import cosine

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

def open_dictionary(dictionary_path):
    if not os.path.exists(dictionary_path):
        raise ValueError("Dictionary not found at {}".format(dictionary_path))
        return
    if dictionary_path.endswith(".json"):
        try:
            with open(dictionary_path, 'r') as fo:
                dictionary = json.load(fo)  # {category: words}
        except Exception:
            raise ValueError("Could not import json dictionary")
    elif dictionary_path.endswith(".dic"):  # traditional LIWC format
        raise ValueError("Dictionary type .dic not supported")
        return
    else:
        raise ValueError("Dictionary type not supported")
    categories, items = zip(*sorted(dictionary.items(), key=lambda x:x[0]))
    return categories, items

def write_file(data, path):

    formatting = path.split('.')[-1]
    if formatting == 'json':
        with open(path, 'w') as fo:
            json.dump(data, fo, indent=4)
    if formatting == 'csv':
        data.to_csv(path)
    if formatting == 'tsv':
        data.to_csv(path, sep='\t')
    if formatting == 'pkl':
        data.to_pickle(path)

class Dataset:
    def __init__(self, source, glove_path=None, mallet_path=None, tokenizer='wordpunct', vocab_size=5000,
            embed='glove', min_token=5, stopwords=None, stem=False,
            lower=True, max_len=100, include_nums=False,
            include_symbols=False, num_topics=100, lda_max_iter=500):
        if isinstance(source, Dataset):
            self = source.copy()
            return
        try:
            self.data = read_file(source)
        except Exception as e:
            print("Exception:", e)
            return
        print("Loaded file with {} documents".format(len(self.data)))
        self.mallet_path = mallet_path
        self.glove_path = glove_path
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
        self.include_nums = include_nums
        self.include_symbols = include_symbols
        self.num_topics = num_topics
        self.lda_max_iter = lda_max_iter

        # destinations for added Dataset objects
        self.features = dict()
        self.feature_names = dict()
        self.__bag_of_words = dict()
        self.targets = dict()
        self.target_names = dict()
        self.weights = dict()

    def copy(self):
        return copy.deepcopy(self)

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

        self.sequence_lengths = list()

        for i, (_, string) in enumerate(self.data[column].iteritems()):
            tokens = self.__tokenize_doc(string)
            self.sequence_lengths.append(len(tokens))
            tokenized[i] = self.__encode_doc(tokens)
        print("Encoded {} docs".format(len(tokenized)))
        print("{} tokens lost to truncation".format(self.__truncate_count))
        print("{} padding tokens added".format(self.__pad_count))
        print("{:.3%} tokens covered by vocabulary of size {}".format((self.__token_count - self.__unk_count) / self.__token_count, len(self.vocab)))
        self.sequence_data = np.array(tokenized)
        self.num_sequences = len(tokenized)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)

    def load_embedding(self, column, embedding_type='glove', embedding_path=None,
            saved_embedding_path=None):
        if not hasattr(self, "vocab"):
            self.__learn_vocab(column)
        load_path = self.glove_path if embedding_type == 'glove' else WORD2VEC
        if saved_embedding_path is not None:
            print("Load from file")
        elif embedding_path is not None:
            load_path = embedding_path

        if embedding_type == 'glove':
            self.embedding, self.embed_dim = self.__read_glove(load_path)
        else:
            raise ValueError("Only glove supported currently")

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
        if "include_symbols" in kwargs:
            self.include_symbols = kwargs["include_symbols"]
        if "include_nums" in kwargs:
            self.include_nums = kwargs["include_nums"]
        if "num_topics" in kwargs:
            self.num_topics = kwargs["num_topics"]
        if "lda_max_iter" in kwargs:
            self.lda_max_iter = kwargs["lda_max_iter"]
        if "dictionary" in kwargs:
            self.dictionary = kwargs["dictionary"]
        if "mallet_path" in kwargs:
            self.mallet_path = kwargs['mallet_path']
        if "glove_path" in kwargs:
            self.glove_path = kwargs['glove_path']

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
        self.mapping = {word: idx for idx, word in enumerate(self.vocab)}

    def __good_doc(self, doc):
        if len(self.tokenizer(doc)) < self.min_token:
            return False
        return True

    def __tokenize_doc(self, doc):
        if type(doc) == list:
            doc = " ".join(doc)
        tokens = self.tokenizer(doc)
        if self.lower:
            tokens = [t.lower() for t in tokens]
        if not self.include_nums and not self.include_symbols:
            tokens = [t for t in tokens if t.isalpha()]
        elif not self.include_nums:
            tokens = [t for t in tokens if not t.isdigit()]
        elif not self.include_symbols:
            tokens = [t for t in tokens if t.isalpha() or t.isdigit()]

        tokens = [t for t in tokens if t not in self.stopwords]
        if self.stem:
            tokens = [stem(w) for w in tokens]
        return tokens

    def __encode_doc(self, doc):
        self.__truncate_count += max(len(doc) - self.max_len, 0)
        unk_idx = self.mapping["<UNK>"]
        pad_idx = self.mapping["<PAD>"]
        encoded = [pad_idx] * self.max_len
        self.__pad_count += max(0, self.max_len - len(doc))
        for i in range(min(self.max_len, len(doc))):  # tokenized
            encoded[i] = self.mapping[doc[i]] if doc[i] in self.mapping else unk_idx
            self.__unk_count += int(encoded[i] == unk_idx)
            self.__token_count += int((encoded[i] != pad_idx) & (encoded[i] != unk_idx))
        return np.array(encoded, dtype=np.int32)

    def encode_targets(self, columns, var_type='categorical', normalize=None,
            encoding='one-hot', reset=False):
        if reset:
            self.data.targets = dict()
            self.data.target_names = dict()

        if not isinstance(columns, list):
            columns = [columns]
        for c in columns:
            if c not in self.data.columns:
                raise ValueError("Column not in Data: {}".format(c))
            if var_type == 'continuous':
                if normalize is not None:
                    pass #TODO: NORM
                else:
                    self.targets[c] = self.data[c].values
                continue
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


    def __batch_indices(self, size, batch_size):
        # produce iterable of (start, end) batch indices
        for i in range(0, size, batch_size):
            start = i
            end = min(size, i + batch_size)
            yield (start, end)

    def batches(self, var_dict, batch_size, test, keep_ratio=None, idx=None):
        feed_dict = dict()

        if idx is None:
            idx = [i for i in range(self.num_sequences)]

        for (s, e) in self.__batch_indices(len(idx), batch_size):
            for var_name in var_dict:
                if var_name == 'word_inputs':
                    feed_dict[var_dict[var_name]] = self.sequence_data[idx[s:e]]
                if var_name == 'sequence_length':
                    feed_dict[var_dict[var_name]] = self.sequence_lengths[idx[s:e]]
                if test:
                    feed_dict[var_dict['keep_ratio']] = 1.0
                    continue  # no labels or loss weights
                if var_name.startswith('target'):
                    name = var_name.replace("target-", "")
                    if name not in self.targets:
                        raise ValueError("Target not in data: {}".format(name))
                    feed_dict[var_dict[var_name]] = self.targets[name][idx[s:e]]
                if var_name.startswith("weights"):
                    name = var_name.replace("weights-", "")
                    if name not in self.weights:
                        raise ValueError("Weights not found in data")
                    feed_dict[var_dict[var_name]] = np.array(self.weights[name])
                if var_name == 'keep_ratio':
                    if keep_ratio is None:
                        raise ValueError("Keep Ratio for RNN Dropout not set")
                    feed_dict[var_dict[var_name]] = keep_ratio
            yield feed_dict

    def get_labels(self, idx, var=None):
        if var is None:
            var = list(self.targets.keys())[0]
        if var not in self.targets:
            raise ValueError("Target not in Dataset object")
        num_classes = len(self.target_names[var])
        if idx is None:
            return self.targets[var], num_classes
        return self.targets[var][idx], num_classes

    def __get_bag_of_words(self, column):
        if not hasattr(self, "vocab"):
            self.__learn_vocab(column)
        encoded = list()
        unk_idx = self.mapping["<UNK>"]

        for doc in self.data[column].values:
            encoded.append(list())
            for token in self.__tokenize_doc(doc):
                id_ = self.mapping[token] if token in self.mapping else None
                if id_ is not None:
                    encoded[-1].append(id_)
        docs = [[(t, doc.count(t)) for t in set(doc)] for doc in encoded]
        id2word = {v:k for k,v in self.mapping.items()}
        return docs, id2word

    def tfidf(self, column):
        self.features["tfidf"] = list()
        if len(self.__bag_of_words) != 0:
            docs, id2word = self.__bag_of_words[column]
        else:
            docs, id2word = self.__get_bag_of_words(column)
        _, words = zip(*id2word.items())
        get_index = {w:i for i, w in enumerate(words)}

        self.feature_names["tfidf"] = words
        transformer = TfidfModel(docs)

        for d in docs:
            sample = transformer[d]
            _doc = np.zeros((len(words)))
            for id_, weight in sample:
                _doc[get_index[id2word[id_]]] = weight
            self.features["tfidf"].append(_doc)
        self.features["tfidf"] = np.array(self.features["tfidf"])

    def lda(self, column, method='mallet', save_model=None, load_model=None):
        if method == 'mallet':
            print("Mallet LDA")
        else:
            raise ValueError("Invalid paramater for LDA.method: {}".format(method))
        tmp_dir = os.path.join(tempfile.gettempdir(), "mallet_lda/")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        if not hasattr(self, "vocab"):
            self.__learn_vocab(column)

        if len(self.__bag_of_words) != 0:
            docs, id2word = self.__bag_of_words[column]
        else:
            docs, id2word = self.__get_bag_of_words(column)
        model = LdaMallet(mallet_path=self.mallet_path,
                          id2word=id2word,
                          prefix=tmp_dir,
                          num_topics=self.num_topics,
                          iterations=self.lda_max_iter,
                          optimize_interval=20)
        model.train(docs)
        doc_topics = list()
        for doc_vec in model.read_doctopics(model.fdoctopics()):
            topic_ids, vecs = zip(*doc_vec)
            doc_topics.append(np.array(vecs))
        self.features["lda"] = np.array(doc_topics)
        self.feature_names["lda"] = model.get_topics()
        return

    def ddr(self, column, dictionary, embed='glove', **kwargs):
        # dictionary can b
        if isinstance(dictionary, str):  # file name
            try:
                dictionary, name = open_dictionary(dictionary)
            except ValueError as e:
                print(e)
                return
        elif isinstance(dictionary, dict):  # {category: [w1, w2...]}
            try:
                sort_dictionary = sorted(dictionary.items(), key=lambda x:x[0])
                dictionary, name = zip(*sort_dictionary)
            except Exception:
                print("Couldn't unpack dictionary")
                return

        if "vocab_size" in kwargs:
            self.__learn_vocab(column, vocab_size=kwargs["vocab_size"])  #TODO
        elif not hasattr(self, "mapping"):
            self.__learn_vocab(column)
        for word_list in name:
            for w in word_list:
                if w not in self.mapping:
                    self.mapping[w] = len(self.mapping)
                    self.vocab.append(w)

        if embed == 'glove':
            embeddings, _ = self.__read_glove(self.glove_path)
        else:
            raise ValueError("Glove only embedding supported")
            return

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
        self.features["ddr"] = features  # type numpy array
        self.feature_names["ddr"] = categories # list of strings

    def __embedding_of_doc(self, doc_string, embeddings, agg='mean', thresh=1):
        tokens = self.__tokenize_doc(doc_string)
        embedded = list()
        for t in tokens:
            if t in self.mapping:
                embedded.append(embeddings[self.mapping[t]])
        if len(embedded) < thresh:
            return np.zeros(embeddings.shape[1])
        if agg == 'mean':
            return np.array(embedded).mean(axis=0)
        else:
            raise ValueError("Aggregation given ({}) not supported")
            return

    def bert(self, some_params):
        print("TODO: implement a featurization based on BERT")

    def __read_glove(self, path, dim=300):
        if not os.path.exists(path):
            raise ValueError("Could not load glove from {}".format(path))
            return
        if path.endswith('.gz'):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'r')

        embeddings = np.random.randn( len(self.mapping), dim)
        found = 0
        for line in f:
            split = line.split()
            idx = len(split) - dim
            type_ = "".join(split[:idx])
            if type_ in self.mapping:
                embeddings[self.mapping[type_]] = np.array(split[idx:],
                        dtype=np.float32)
                found += 1
        print("Found {}/{} of vocab in {}".format(found, len(self.mapping),
            path))
        f.close()
        return embeddings, dim

    def save_embeddings(self, dir_):
        if not hasattr(self, "embedding"):
            raise ValueError("No embedding found in Dataset object")
            return
        np.save(os.path.join(dir_, "embedding.npy"), self.embedding)
        write_file(self.mapping, os.path.join(dir_, "vocab.json"))
