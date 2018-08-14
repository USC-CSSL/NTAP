import numpy as np
import tensorflow as tf
import pandas as pd
import os, math
import operator
from neural.LSTM import LSTM
from neural.CNN import CNN
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from nltk import tokenize as nltk_token
#from set_globals import glove_path, word2vec_path
from random import randint
#from tokenization.tokenizers import wordpunc_tokenize, happiertokenize, tweettokenize
#toks = {'happier': happiertokenize,
#        'wordpunc': wordpunc_tokenize,
#        'tweet': tweettokenize}

class Neural:
    def __init__(self, params):
        self.params = params
        for key in params:
            setattr(self, key, params[key])
        if self.word_embedding == 'glove':
            #self.embeddings_path = glove_path
            self.embeddings_path = os.environ['GLOVE_PATH']
        else:
            #self.embeddings_path = word2vec_path
            self.embeddings_path = os.environ['WORD2VEC_PATH']

    def tokenize_data(self, corpus):
        #sent_tokenizer = toks[self.params["tokenize"]]
        tokenized_corpus = [nltk_token.WordPunctTokenizer().tokenize(sent.lower()) for sent in corpus]
        return tokenized_corpus

    def learn_vocab(self, corpus):
        print("Learning vocabulary of size %d" % (self.vocab_size))
        tokens = dict()
        for sent in corpus:
            for token in sent:
                if token in tokens:
                    tokens[token] += 1
                else:
                    tokens[token] = 1
        words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
        self.vocab = list(words[:self.vocab_size]) + ["<unk>", "<pad>"]

    def tokens_to_ids(self, corpus, learn_max=True):
        print("Converting corpus of size %d to word indices based on learned vocabulary" % len(corpus))
        if self.vocab is None:
            raise ValueError("learn_vocab before converting tokens")

        mapping = {word: idx for idx, word in enumerate(self.vocab)}
        unk_idx = self.vocab.index("<unk>")
        for i in range(len(corpus)):
            row = corpus[i]
            for j in range(len(row)):
                try:
                    corpus[i][j] = mapping[corpus[i][j]]
                except:
                    corpus[i][j] = unk_idx
        if learn_max:
            self.max_length = max([len(line) for line in corpus])
        return corpus


    def build(self):
        if self.pretrain:
            self.load_glove()
        if self.model == "LSTM" or self.model == "BiLSTM":
            self.nn = LSTM(self.params, self.max_length, self.vocab, self.embeddings)
        else:
            self.nn = CNN()
        self.nn.build()

    def graph(self, vectors, labels):
        pca = PCA(n_components=2)
        vec_components = pca.fit_transform(vectors)
        df = pd.DataFrame(data=vec_components, columns=['component 1', 'component 2'])
        finalDf = pd.concat([df, labels], axis=1)
        return finalDf


    def run_model(self, X, y, labels):
        vectors = self.nn.get_vectors(self.get_batches(X, y))
        self.graph(vectors, labels)


    def cv_model(self, X, y):
        kf = KFold(n_splits=self.params["k_folds"], shuffle=True, random_state=self.random_seed)
        predictions = dict()
        vectors = dict()
        indices = dict()
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # train_idx, test_idx = np.array(indices)[train_index], np.array(indices)[test_index]
            vectors_iter, prediction_iter = self.nn.run_model(self.get_batches(X_train, y_train) ,
                                                                  self.get_batches(X_test, y_test))
            predictions[idx] = prediction_iter
            vectors[idx] = vectors_iter
            indices[idx] = test_idx


    def load_embeddings(self):
        if self.word_embedding == 'glove':
            self.load_glove()

    def load_glove(self):
        if not os.path.isfile(self.embeddings_path):
            raise IOError("You're trying to access an embeddings file that doesn't exist")
        self.embeddings = dict()
        with open(self.embeddings_path, 'r') as fo:
            glove = dict()
            for line in fo:
                tokens = line.split()
                embedding = np.array(tokens[len(tokens) - self.embedding_size:], dtype=np.float32)
                token = "".join(tokens[:len(tokens) - self.embedding_size])
                glove[token] = embedding
                # stdout.write("\r{} tokens read from file".format(len(glove)))
                # stdout.flush
        unk_embedding = np.random.rand(self.embedding_size) * 2. - 1.
        if self.vocab is None:
            print("Error: Build vocab before loading GloVe vectors")
            exit(1)
        not_found = 0
        for token in self.vocab:
            try:
                self.embeddings[token] = glove[token]
            except KeyError:
                not_found += 1
                self.embeddings[token] = unk_embedding
        print(" %d tokens not found in GloVe embeddings" % (not_found))
        # self.embeddings = np.array(collections.OrderedDict(sorted(self.embeddings.items())).values())
        self.embeddings = np.array(list(self.embeddings.values()))

    def get_batches(self, corpus_ids, labels=None):
        batches = []
        for idx in range(len(corpus_ids) // self.batch_size + 1):
            labels_batch = labels[idx * self.batch_size: min((idx + 1) * self.batch_size,
                                                        len(labels))] if labels is not None else []
            text_batch = corpus_ids[idx * self.batch_size: min((idx + 1) * self.batch_size, len(corpus_ids))]
            lengths = np.array([len(line) for line in text_batch])
            text_batch = self.padding(text_batch)
            batches.append((np.array([np.array(line) for line in text_batch]), lengths, np.array(labels_batch)))
        return batches

    def padding(self, corpus):
        padd_idx = self.vocab.index("<pad>")
        for i in range(len(corpus)):
            corpus[i] = corpus[i][:min(len(corpus[i]), self.max_length) - 1]
            while len(corpus[i]) < self.max_length:
                corpus[i].append(padd_idx)
        return corpus