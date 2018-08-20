import numpy as np
import tensorflow as tf
import pandas as pd
import os, math
import operator
from sys import stdout
import collections
from tensorflow.contrib.layers import fully_connected
from sklearn.model_selection import train_test_split, KFold
from random import randint


class CNN():
    def __init__(self, params, max_length, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        self.max_length = max_length
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=self.train_embedding, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocab), self.embedding_size])
            embedded_chars_expanded = tf.expand_dims(embedding_placeholder, -1)
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
            embedded_chars_expanded = tf.expand_dims(embedding_placeholder, -1)
        return embedded_chars_expanded


    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


    def build(self):
        self.embedding_placeholder = self.build_embedding()

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, self.max_length], name="inputs")
        embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        pooled_outputs = list()


        self.keep_prob = tf.placeholder(tf.float32)

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(embed, W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            pooled = tf.nn.max_pool(relu, ksize=[1, self.max_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        self.task_outputs = dict()
        for target in self.target_cols:
            y = tf.placeholder(tf.int64, [None], name=target)
            self.task_outputs[target] = y

        self.logits = dict()
        self.predictions = dict()
        self.loss = dict()
        self.accuracy = dict()
        self.xentropy = dict()
        self.drop_out = dict()
        self.predicted_label = dict()

        for target in self.target_cols:
            self.predictions[target] = fully_connected(drop, self.n_outputs)

            self.xentropy[target] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.task_outputs[target],
                                                                                   logits=self.predictions[target])
            self.loss[target] = tf.reduce_mean(self.xentropy[target])

            self.predicted_label[target] = tf.argmax(self.predictions[target], 1)
            self.accuracy[target] = tf.reduce_mean(
                tf.cast(tf.equal(self.predicted_label[target], self.task_outputs[target]), tf.float32))

        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        if self.loss == "Mean":
            self.joint_loss = sum(self.loss.values()) / len(self.target_cols)
        else:#elif self.loss == "Weighted":
            for target in self.target_cols:
                self.loss[target] *= (1 - self.accuracy[target])
            self.joint_loss = sum(self.loss.values()) / len(self.target_cols)

        self.training_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)

    def splitY(self, y_data, feed_dict):
        for i in range(len(self.target_cols)):
            feed_dict[self.task_outputs[self.target_cols[i]]] = y_data[:, i]
        return feed_dict


    def get_vectors(self, batches):
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            epoch = 0
            while True:
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for (X_batch, X_len, y_batch) in batches:
                    feed_dict = self.splitY(y_batch, {self.train_inputs: X_batch,
                                                      self.keep_prob: self.keep_ratio})
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val = self.sess.run(
                        [self.training_op, self.joint_loss], feed_dict=feed_dict)
                    epoch_loss += loss_val
                    acc_train += self.joint_accuracy.eval(feed_dict=feed_dict)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)))
                if epoch > 500:
                    break
        vectors = list()
        for (X_batch, X_len, y_batch) in batches:
            feed_dict = {self.train_inputs: X_batch, 
                self.keep_prob: self.keep_ratio}
            if self.pretrain:
                feed_dict[self.embedding_placeholder] = self.my_embeddings
            vector = self.sess.run(self.last,
                                       feed_dict=feed_dict)

            vectors.extend(vector)
        vectors = np.array(vectors)
        vectors.dump(os.environ['FEAT_PATH'] + "/vectors.pkl")
        return vectors

    def run_model(self, batches, test_batches):
        init = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            test_predictions = {target: np.array([]) for target in self.target_cols}
            while True:
                ## Train
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for (X_batch, X_len, y_batch) in batches:
                    feed_dict = self.splitY(y_batch, {self.train_inputs: X_batch,
                                                      self.keep_prob: self.keep_ratio})
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val= self.sess.run([self.training_op, self.joint_loss], feed_dict= feed_dict)

                    acc_train += self.joint_accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                acc_test = 0
                for (X_batch, X_len, y_batch) in test_batches:
                    feed_dict = self.splitY(y_batch, {self.train_inputs: X_batch,
                                                        self.keep_prob: 1})
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    acc_test += self.joint_accuracy.eval(feed_dict=feed_dict)
                    if epoch == self.epochs:
                        for target in self.target_cols:
                            test_predictions[target] = np.append(test_predictions[target], self.predicted_label[target].eval(feed_dict=feed_dict))

                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))

                if epoch == self.epochs:
                    test_predictions = np.transpose(np.array([test_predictions[target] for target in self.target_cols]))
                    break
            #save_path = saver.save(self.sess, "/tmp/model.ckpt")
        return test_predictions, acc_test / float(len(test_batches))