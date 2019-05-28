#NeuralModel.py has implementation of a super-class "NeuralModel" and a bunch of sub-classes extending it like "LSTM", "LSTM_feat", "CNN", "ATTN" and "ATTN_feat"

from methods.neural.nn import *

class baseModel():
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        self.all_params = all_params
        self.neural_params = all_params['neural_params']
        self.vocab = vocab
        for key in self.neural_params:
            setattr(self, key, self.neural_params[key])
        self.max_length = max_length
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def buildEmbedding(self, pretrain, train_embedding, embedding_size, vocab_size, expand_dims):
        if pretrain:
            embeddings_variable = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                                     trainable=train_embedding, name="W")
        else:
            embeddings_variable = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [vocab_size, embedding_size], -1, 1),
                                                    dtype=tf.float32)
        if expand_dims==True:
            embeddings_variable = tf.expand_dims(embeddings_variable,-1)
        return embeddings_variable

    def buildOptimizer(self):
        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.target_cols:
            self.loss[target], self.predict[target], self.accuracy[target] = self.buildPredictor(self.state,
                                                                                  self.n_outputs,
                                                                                  self.weights[target],
                                                                                  self.task_outputs[target])
        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        self.joint_loss = sum(self.loss.values())
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)

    def build_CNN(self, input, filter_sizes, num_filters, keep_ratio):
        pooled_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, int(input.get_shape()[2]), 1, num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        output = tf.nn.dropout(h_pool_flat, keep_ratio)
        return output

    # What is the use of this?
    def dropPadding(self, output, length):
        relevant = tf.gather(output, length, axis = 1)
        return relevant

    def dynamic_rnn(self, cell, model, hidden, keep_prob, num_layers, embed, sequence_length):
        if model[:4] == "LSTM":
            network = self.multi_GRU(cell, hidden, keep_prob, num_layers)
            rnn_outputs, state = tf.nn.dynamic_rnn(network, embed,
                                                   dtype=tf.float32, sequence_length=sequence_length)
            if cell == "GRU":
                state = state[0]
            else:
                state = state[0].h
        else:
            f_network = self.multi_GRU(cell, hidden, keep_prob, num_layers)
            b_network = self.multi_GRU(cell, hidden, keep_prob, num_layers)
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(f_network, b_network, embed,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length)
            fw_outputs, bw_outputs = bi_outputs
            fw_states, bw_states = bi_states
            rnn_outputs = tf.concat([fw_outputs, bw_outputs], 2)
            if cell == "GRU":
                state = tf.concat([fw_states[0], bw_states[0]], 1)
            else:
                state = tf.concat([fw_states[0].h, bw_states[0].h], 1)
        return rnn_outputs, state

    def initialise(self):
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = self.buildEmbedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab), self.expand_dims)
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.task_outputs = self.multi_outputs(self.target_cols)
        self.weights = self.weightPlaceholder(self.target_cols)
        self.keep_prob = tf.placeholder(tf.float32)

    def multi_GRU(self, cell, hidden_size, keep_ratio, num_layers):
        if cell == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
        elif cell == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_ratio)
        network = tf.contrib.rnn.MultiRNNCell([cell_drop] * num_layers)
        return network

    def multi_outputs(self,target_cols):
        outputs = dict()
        for target in target_cols:
            y = tf.placeholder(tf.int64, [None], name=target)
            outputs[target] = y
        return outputs

    def buildPredictor(self, hidden, n_outputs, weights, task_outputs):
        logits = tf.layers.dense(hidden, n_outputs)

        weight = tf.gather(weights, task_outputs)
        xentropy = tf.losses.sparse_softmax_cross_entropy(labels=task_outputs,
                                                          logits=logits,
                                                          weights=weight)
        loss = tf.reduce_mean(xentropy)
        predicted_label = tf.argmax(logits, 1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_label, task_outputs), tf.float32))

        return loss, predicted_label, accuracy

    def predictModel(self, batches, data_batches, weights, savedir):
        return execute_prediction_process(self, batches, data_batches, weights, savedir, self.all_params)

    def trainModel(self, batches, test_batches, weights):
        return execute_training_process(self, batches, test_batches, weights, self.all_params)

    def weightPlaceholder(self, target_cols):
        weights = dict()
        for target in target_cols:
            weights[target] = tf.placeholder(tf.float64, [None], name=target + "_w")
        return weights
