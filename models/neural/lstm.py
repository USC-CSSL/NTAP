import numpy as np
import tensorflow as tf
import pandas as pd
import os
import operator
from sys import stdout
import collections

class LSTM:
    def __init__(self, hidden_size, num_layers, learning_rate, batch_size,
                 vocab_size=10000, embedding_size=300,
                 dropout_ratio=None, pretrain=False, use_pretraining=False):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.vocab_size = vocab_size
        print(self.vocab_size)
        self.embedding_size = embedding_size

    def learn_vocab(self, corpus):
        tokens = dict()
        for sent in corpus:
            for token in sent:
                if token in tokens:
                    tokens[token] += 1
                else:
                    tokens[token] = 1
        words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
        self.vocab = list(words[:self.vocab_size]) + ["<unk>", "<pad>"]

    def tokens_to_ids(self, corpus, learn_max = False):
        if self.vocab is None:
            raise ValueError("learn_vocab before converting tokens")
        
        mapping = {word:idx for idx, word in enumerate(self.vocab)}
        unk_idx = self.vocab.index("<unk>")
        for i in range(len(corpus)):
            row = corpus[i]
            if len(row) > 400:
                print(i)
            for j in range(len(row)):
                try:
                    corpus[i][j] = mapping[corpus[i][j]]
                except:
                    corpus[i][j] = unk_idx
        #print("Max length is ", max([len(line) for line in corpus]))
        if learn_max:
            self.max_length = max([len(line) for line in corpus])
        return corpus

    def get_batches(self, corpus_ids, labels):
        # given list of lists of token ids, return iterable of lists of ids
        #unique, counts = np.unique(labels, return_counts=True)

        #print("Test batch has %d 0s and %d 1s." % (dict(zip(unique, counts))[0], dict(zip(unique, counts))[1]))
        batches = []
        if self.vocab is None:
            raise ValueError("learn_vocab before converting tokens")
        for idx in range( len(corpus_ids) // self.batch_size):
            labels_batch = labels[idx * self.batch_size: min((idx + 1) * self.batch_size, len(labels) - 1)]
            text_batch = corpus_ids[idx * self.batch_size: min((idx + 1) * self.batch_size, len(corpus_ids) - 1)]
            lengths = np.array([len(line) for line in text_batch])
            #max_length = np.max(lengths)
            #max length should be the largest size of the whole data, not the batch
            text_batch = self.padding(text_batch)
            batches.append((np.array([np.array(line) for line in text_batch]), lengths, np.array(labels_batch)))
        return batches

    def padding(self, corpus):
        padd_idx = self.vocab.index("<pad>")
        for i in range(len(corpus)):
            corpus[i] = corpus[i][:81]
            while len(corpus[i]) < self.max_length:
                corpus[i].append(padd_idx)
        return corpus


    def build(self):
        embed_size = 300
        n_outputs = 6  # 0 1 2 3 4 5
        if not self.use_pretrain:
            embeddings = tf.get_variable("embedding", 
                                initializer=tf.random_uniform([vocab_size, embed_size], -1, 1),
                                dtype=tf.float32)
            train_inputs = tf.placeholder(tf.int32, shape=[None])
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
        if dropout_ratio is not None:
            cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_ratio)
            self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)
        else:
            self.network = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers)

        y = tf.placeholder(tf.int32, [None])
        
        self.rnn_outputs, self.states = tf.nn.dynamic_rnn(self.network, embed, 
                                            dtype=tf.float32, sequence_length=source_sequence_length)
        logits = fully_connected(self.states, n_outputs, activation_fn=None)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def load_glove(self, path, seed, embedding_size=300):
        if not os.path.isfile(path):
            raise IOError("You're trying to access a GloVe embeddings file that doesn't exist")
        self.embeddings = dict()
        with open(path, 'r') as fo:
            glove = dict()
            for line in fo:
                tokens = line.split()
                embedding = np.array(tokens[len(tokens) - embedding_size:], dtype=np.float32)
                token = "".join(tokens[:len(tokens) - embedding_size])
                glove[token] = embedding
                stdout.write("\r{} tokens read from file".format(len(glove)))
                stdout.flush
        unk_embedding = np.random.rand(embedding_size) * 2. - 1.
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
        #self.embeddings = np.array(collections.OrderedDict(sorted(self.embeddings.items())).values())
        self.embeddings = np.array(list(self.embeddings.values()))

        print(self.embeddings.shape)
        return self.embeddings
