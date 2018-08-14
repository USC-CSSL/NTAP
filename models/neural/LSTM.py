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


class LSTM():
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
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder


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

        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        if self.cell == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
        elif self.cell == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)


        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        self.task_outputs = dict()
        for target in self.target_cols:
            y = tf.placeholder(tf.int64, [None], name=target)
            self.task_outputs[target] = y

        if self.model == "BiLSTM":
            rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(network, network, embed,
                                                                  dtype=tf.float32, sequence_length=self.sequence_length)
            fw_outputs, bw_outputs = rnn_outputs
            fw_state, bw_state = states

            fw_last = self.last_relevant(fw_outputs, self.sequence_length)
            bw_last = self.last_relevant(fw_outputs, self.sequence_length)

            last = tf.concat([fw_last, bw_last], 1)
        else:
            rnn_outputs, states = tf.nn.dynamic_rnn(self.network, embed,
                                                              dtype=tf.float32, sequence_length=self.sequence_length)
            states_concat = tf.concat(axis=1, values=states)
            last = self.last_relevant(rnn_outputs, self.sequence_length)

        self.logits = dict()
        self.predictions = dict()
        self.loss = dict()
        self.accuracy = dict()
        self.xentropy = dict()
        self.drop_out = dict()

        for target in self.target_cols:
            self.logits[target] = fully_connected(last, math.floor(self.hidden_size / 2), activation_fn=tf.nn.sigmoid)
            self.drop_out[target] = tf.contrib.layers.dropout(self.logits[target], self.keep_prob)
            # task_logits[target] = drop_out

            self.predictions[target] = fully_connected(self.drop_out[target], self.n_outputs)

            self.xentropy[target] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.task_outputs[target],
                                                                                   logits=self.predictions[target])
            self.loss[target] = tf.reduce_mean(self.xentropy[target])

            self.accuracy[target] = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.predictions[target], 1), self.task_outputs[target]), tf.float32))

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
                                                      self.sequence_length: X_len,
                                                      self.keep_prob: self.keep_ratio,
                                                      self.embedding_placeholder: self.my_embeddings if self.pretrain else None})
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
            vector = self.sess.run(self.last,
                                       feed_dict={self.train_inputs: X_batch, self.sequence_length: X_len,
                                                  self.keep_prob: self.keep_ratio,
                                                  self.embedding_placeholder: self.my_embeddings if self.pretrain else None})
            vectors.extend(vector)
        vectors = np.array(vectors)
        vectors.dump(os.environ['FEAT_PATH'] + "/vectors.pkl")
        return vectors

    def run_model(self, batches, test_batches):
        init = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            init.run()
            epoch = 0
            while True:
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for (X_batch, X_len, y_batch) in batches:
                    feed_dict = self.splitY(y_batch, {self.train_inputs: X_batch,
                                                      self.sequence_length: X_len,
                                                      self.keep_prob: self.keep_ratio,
                                                      self.embedding_placeholder: self.my_embeddings if self.pretrain else None})
                    _, loss_val, predictions_= self.sess.run([self.training_op, self.joint_loss, self.predictions], feed_dict= feed_dict)
                    epoch_loss += loss_val
                    #print(predictions_["hate"])
                    acc_train += self.joint_accuracy.eval(feed_dict= feed_dict)

                acc_test = 0
                for (X_batch, X_len, y_batch) in test_batches:
                    acc_test += self.joint_accuracy.eval(
                        feed_dict=self.splitY(y_batch, {self.train_inputs: X_batch, self.sequence_length: X_len,
                                                        self.keep_prob: 1,
                                                        self.embedding_placeholder: self.my_embeddings if self.pretrain else None}))

                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))

                if epoch > 500:
                    break
            #acc.append(acc_test / float(len(test_batches)))
            #save_path = saver.save(self.sess, "/tmp/model.ckpt")
"""
            results = dict()
            for (text_batch, lengths, _) in self.get_batches(X):
                output = self.sess.run(self.predictions,
                                       feed_dict={self.train_inputs: text_batch, self.sequence_length: lengths,
                                                  self.keep_prob: 1,
                                                  self.embedding_placeholder: self.my_embeddings if self.pretrain else None})
                for target in self.target_cols:
                    results.setdefault(target, []).extend(list(np.argmax(output[target], 1)))
            results["index"] = indices
            re_dataframe = pd.DataFrame(results)

            for target in self.target_cols:
                print("Analyzing results of", target)
                true = dict()
                false = dict()
                all = dict()
                precision = dict()
                f1 = dict()
                recall = dict()
                checked = list()
                for i in range(self.n_outputs):
                    true[i] = 0
                    all[i] = 0
                    false[i] = 0

                for idx in test_idx:
                    if idx in checked:
                        continue
                    checked.append(idx)
                    if (re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target] ==
                            annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]).any():
                        true[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] += 1
                    else:
                        false[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] += 1
                    all[annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]] += 1

                for i in range(n_outputs):
                    precision[i] = float(true[i]) / float(true[i] + false[i]) if true[i] + false[
                        i] != 0 else 0
                    recall[i] = float(true[i]) / float(all[i]) if all[i] != 0 else 0
                    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + \
                                                                                         recall[
                                                                                             i] > 0 else 0
                none.append(f1[0])
                qes.append(f1[1])
                prob.append(f1[2])
                print("none", f1[0], "ques", f1[1], "prob", f1[2])

    print(float(sum(none)) / float(len(none)))
    print(float(sum(qes)) / float(len(qes)))
    print(float(sum(prob)) / float(len(prob)))
    print(float(sum(acc)) / float(len(acc)))
    results = pd.DataFrame(
        {'none': none,
         'ques': qes,
         'prob': prob,
         'acc': acc
         })
    results.to_pickle("results.pkl")
"""