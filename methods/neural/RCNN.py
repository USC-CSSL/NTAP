import numpy as np
import tensorflow as tf
import os, copy
from tensorflow.contrib.layers import fully_connected
from neural.utils import  *
from tensorflow.python.ops import array_ops

class RCNN():
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

    def build(self):
        self.embedding_placeholder = self.build_embedding()
        self.sequence_length = tf.placeholder(tf.int32, [None])

        self.keep_prob = tf.placeholder(tf.float32)

        if self.cell == "LSTM":
            f_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, reuse=tf.AUTO_REUSE)
            b_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, reuse=tf.AUTO_REUSE)
        elif self.cell == "GRU":
            f_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            b_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        f_cell_drop = tf.contrib.rnn.DropoutWrapper(f_cell, input_keep_prob=self.keep_prob)
        b_cell_drop = tf.contrib.rnn.DropoutWrapper(b_cell, input_keep_prob=self.keep_prob)

        forward_network = tf.contrib.rnn.MultiRNNCell([f_cell_drop] * self.num_layers)
        backward_network = tf.contrib.rnn.MultiRNNCell([b_cell_drop] * self.num_layers)

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, self.max_length], name="inputs")

        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.forward_first = tf.zeros((tf.shape(self.embed)[0], 1, self.embedding_size))
        self.backward_first = tf.zeros((tf.shape(self.embed)[0], 1, self.embedding_size))

        self.forward_rnn_outputs, _ = tf.nn.dynamic_rnn(forward_network, tf.concat([self.forward_first, self.embed], 1), dtype=tf.float32, sequence_length=self.sequence_length)

        embed_rev = array_ops.reverse_sequence(self.embed, seq_lengths=self.sequence_length, seq_dim=1)
        backward_rnn_temp, _ = tf.nn.dynamic_rnn(backward_network, tf.concat([self.backward_first, embed_rev], 1), dtype=tf.float32, sequence_length=self.sequence_length)

        self.backward_rnn_outputs = array_ops.reverse_sequence(backward_rnn_temp, seq_lengths=self.sequence_length, seq_dim=1)

        self.forward_embed_backward = tf.expand_dims(tf.concat([tf.slice(self.forward_rnn_outputs, [0, 0, 0], [tf.shape(self.embed)[0], tf.shape(self.embed)[1], self.hidden_size]), self.embed, tf.slice(self.backward_rnn_outputs, [0, 0, 0], [tf.shape(self.embed)[0], tf.shape(self.embed)[1], self.hidden_size])], 2), -1)

        # As in CNN, we keep the paddings..
        #self.last_forward = self.drop_padding(self.forward_rnn_outputs, self.sequence_length)


        pooled_outputs = list()

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, int(self.forward_embed_backward.get_shape()[2]), 1, self.num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(self.forward_embed_backward, W, strides=[1, 1, 1, 1], padding="VALID")
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

        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)


    def splitY(self, y_data, feed_dict):
        for i in range(len(self.target_cols)):
            feed_dict[self.task_outputs[self.target_cols[i]]] = y_data[:, i]
        return feed_dict

    def drop_padding(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        relevant = tf.gather(output, length, axis = 1)
        return relevant

    def get_vectors(self, batches):
        tf.reset_default_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            epoch = 0
            while True:
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for (X_batch, X_len, y_batch) in batches:
                    feed_dict = splitY(self, y_batch, {self.train_inputs: X_batch,
                                                      self.keep_prob: self.keep_ratio})
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val = self.sess.run(
                        [self.training_op, self.joint_loss], feed_dict=feed_dict)
                    epoch_loss += loss_val
                    acc_train += self.joint_accuracy.eval(feed_dict=feed_dict)
                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)))
                if epoch > 300:
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
                    feed_dict = splitY(self, y_batch, {self.train_inputs: X_batch,
                                                       self.sequence_length: X_len,
                                                      self.keep_prob: self.keep_ratio})
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val, predictions_= self.sess.run([self.training_op, self.joint_loss, self.predictions], feed_dict= feed_dict)
                    acc_train += self.joint_accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                acc_test = 0
                for (X_batch, X_len, y_batch) in test_batches:
                    feed_dict = splitY(self, y_batch, {self.train_inputs: X_batch,
                                                       self.sequence_length: X_len,
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