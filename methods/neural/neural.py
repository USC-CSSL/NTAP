import numpy as np
import tensorflow as tf
import pandas as pd
import os, math
from methods.neural.LSTM import LSTM
from methods.neural.CNN import CNN
from methods.neural.RCNN import RCNN
from methods.neural.Attn import ATTN
from methods.neural.Attn_feat import ATTN_feat
from methods.neural.LSTM_feat import LSTM_feat
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from collections import Counter

class Neural:
    def __init__(self, params, vocab):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.word_embedding == 'glove':
            self.embeddings_path = self.glove_path
            #self.embeddings_path = os.environ['GLOVE_PATH']
        else:
            self.embeddings_path = self.word2vec_path
            #self.embeddings_path = os.environ['WORD2VEC_PATH']


    def build(self):
        if self.pretrain:
            self.load_glove()
        else:
            self.embeddings = None

        if self.model == "LSTM" or self.model == "BiLSTM":
            self.nn = LSTM(self.params, self.max_length, self.vocab, self.embeddings)
        elif self.model == "CNN":
            if self.pretrain:
                self.embeddings.reshape(self.embeddings.shape[0], self.embeddings.shape[1], 1)
            self.nn = CNN(self.params, self.max_length, self.vocab, self.embeddings)
        elif self.model == "RCNN":
            self.nn = RCNN(self.params, self.vocab, self.embeddings)
        elif self.model == "ATTN":
            self.nn = ATTN(self.params, self.vocab, self.embeddings)
        elif self.model == "ATTN_feat":
            self.nn = ATTN_feat(self.params, self.vocab, self.embeddings)
        elif self.model == "LSTM_feat":
            self.nn = LSTM_feat(self.params, self.vocab, self.embeddings)
        self.nn.build()

    def graph(self, vectors, labels):
        pca = PCA(n_components=2)
        vec_components = pca.fit_transform(vectors)
        df = pd.DataFrame(data=vec_components, columns=['component 1', 'component 2'])
        finalDf = pd.concat([df, labels], axis=1)
        return finalDf


    def run_model(self, X, y, data, weights, savedir):
        self.nn.predict_labels(self.get_batches(X, y), self.get_batches(data), weights, savedir)


    def cv_model(self, X, y, weights, savedir):
        kf = KFold(n_splits=self.params["kfolds"], shuffle=True, random_state=self.random_seed)
        f1s = {target: list() for target in self.target_cols}
        ps = {target: list() for target in self.target_cols}
        rs = {target: list() for target in self.target_cols}
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print("Cross validation, iteration", idx + 1)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            f1_scores, precision, recall = self.nn.run_model(self.get_batches(X_train, y_train),
                                          self.get_batches(X_test, y_test), weights)
            for target in self.target_cols:
                f1s[target].append(f1_scores[target])
                ps[target].append(precision[target])
                rs[target].append(recall[target])
        for target in self.target_cols:
            print("Overall F1 for", target, ":", sum(f1s[target]) / self.params["kfolds"])
            print("Overall Precision for", target, ":", sum(ps[target]) / self.params["kfolds"])
            print("Overall Recall for", target, ":", sum(rs[target]) / self.params["kfolds"])
        #pd.DataFrame.from_dict(f1s).to_csv(savedir + "/" + ".".join(t for t in self.target_cols) + ".csv")

    def load_embeddings(self):
        if self.word_embedding == 'glove':
            self.load_glove()

    def load_glove(self):
        if not os.path.isfile(self.glove_path):
            raise IOError("You're trying to access an embeddings file that doesn't exist")
        self.embeddings = dict()
        with open(self.embeddings_path, 'r') as fo:
            glove = dict()
            for line in fo:
                tokens = line.split()
                embedding = np.array(tokens[len(tokens) - self.embedding_size:], dtype=np.float32)
                token = "".join(tokens[:len(tokens) - self.embedding_size])
                glove[token] = embedding
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
        self.embeddings = np.array(list(self.embeddings.values()))

    def get_batches(self, corpus_ids, labels=None, features=None, padding=True):
        batches = []
        for idx in range(len(corpus_ids) // self.batch_size + 1):
            labels_batch = labels[idx * self.batch_size: min((idx + 1) * self.batch_size,
                                len(labels))] if labels is not None else []

            text_batch = corpus_ids[idx * self.batch_size: min((idx + 1) * self.batch_size,
                                len(corpus_ids))]

            features_batch = features[idx * self.batch_size: min((idx + 1) * self.batch_size,
                                len(features))] if features is not None else []

            lengths = np.array([len(line) for line in text_batch])
            if padding:
                text_batch = self.padding(text_batch)
            if len(text_batch) > 0:
                batches.append({"text": np.array([np.array(line) for line in text_batch]),
                                "sent_lens": lengths,
                                "label": np.array(labels_batch),
                                "feature": np.array(features_batch)})
        return batches

    def padding(self, corpus):
        padd_idx = self.vocab.index("<pad>")
        for i in range(len(corpus)):
            while len(corpus[i]) < max(len(sent) for sent in corpus):
                corpus[i].append(padd_idx)
        return corpus