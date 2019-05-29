import numpy as np
import tensorflow as tf
import pandas as pd
import os, math
from helperFunctions import get_batches, loadEmbeddings
from methods.neural.LSTM import LSTM
from methods.neural.LSTM_feat import LSTM_feat
from methods.neural.CNN import CNN
from methods.neural.ATTN import ATTN
from methods.neural.ATTN_feat import ATTN_feat
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from collections import Counter, defaultdict
import statistics

class Neural:
    def __init__(self, all_params, vocab):
        self.all_params = all_params
        self.neural_params = self.all_params['neural_params']
        self.vocab = vocab
        self.glove_path = all_params['path']['glove_path']
        self.word2vec_path = all_params['path']['word2vec_path']
        for key in self.neural_params:
            setattr(self, key, self.neural_params[key])
        if self.word_embedding == 'glove':
            self.embeddings_path = self.glove_path
        else:
            self.embeddings_path = self.word2vec_path


    def build(self):
        if self.pretrain:
            self.embeddings = loadEmbeddings(self.word_embedding, self.vocab, self.glove_path, self.embeddings_path, self.embedding_size)
        else:
            self.embeddings = None

        if self.model == "LSTM" or self.model == "BiLSTM":
            self.nn = LSTM(self.all_params, self.max_length, self.vocab, self.embeddings)
        elif self.model == "CNN":
            if self.pretrain:
                self.embeddings = self.embeddings.reshape(self.embeddings.shape[0], self.embeddings.shape[1], 1)
            self.nn = CNN(self.all_params, self.max_length, self.vocab, self.embeddings)
        elif self.model == "ATTN":
            self.nn = ATTN(self.all_params,  self.max_length, self.vocab, self.embeddings)
        elif self.model == "ATTN_feat":
            self.nn = ATTN_feat(self.all_params, self.max_length, self.vocab, self.embeddings)
        elif self.model == "LSTM_feat":
            self.nn = LSTM_feat(self.all_params, self.max_length, self.vocab, self.embeddings)
        self.nn.build()

    def trainModelUsingCV(self, X, y, weights, savedir, features):
        kf = KFold(n_splits=self.neural_params["kfolds"], shuffle=True, random_state=self.random_seed)
        f1s = {target: list() for target in self.target_cols}
        ps = {target: list() for target in self.target_cols}
        rs = {target: list() for target in self.target_cols}
        scores = defaultdict(lambda: defaultdict(list))
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print("Cross validation, iteration", idx + 1)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if self.nn.feature:
                print(features.shape)
                feat_train, feat_test = features[train_idx], features[test_idx]
            else:
                feat_train, feat_test = list(), list()
            f1_scores, precision, recall = self.nn.trainModel(get_batches(self.batch_size, self.vocab, X_train, y_train, feat_train), get_batches(self.batch_size, self.vocab, X_test, y_test, feat_test), weights)

            for target in self.target_cols:
                scores[target]['f1'].append(f1_scores[target])
                scores[target]['precision'].append(precision[target])
                scores[target]['recall'].append(recall[target])
                # f1s[target].append(f1_scores[target])
                # ps[target].append(precision[target])
                # rs[target].append(recall[target])
        for k, v in scores.items():
            pd.DataFrame.from_dict(v).to_csv(savedir + "/" + k + ".csv")
        # for target in self.target_cols:
        #     print("Overall F1 for", target, ":", sum(f1s[target]) / self.neural_params["kfolds"])
        #     print("Standard Deviation:", statistics.stdev(f1s[target]))
        #     print("Overall Precision for", target, ":", sum(ps[target]) / self.neural_params["kfolds"])
        #     print("Standard Deviation:", statistics.stdev(ps[target]))
        #     print("Overall Recall for", target, ":", sum(rs[target]) / self.neural_params["kfolds"])
        #     print("Standard Deviation:", statistics.stdev(rs[target]))
        #pd.DataFrame.from_dict(f1s).to_csv(savedir + "/" + ".".join(t for t in self.target_cols) + ".csv")


    def predictModel(self, X, y, data, weights, savedir):
        self.nn.predictModel(get_batches(self.batch_size, self.vocab, X, y), get_batches(self.batch_size, self.vocab, data), weights, savedir)
