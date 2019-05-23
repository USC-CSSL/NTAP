#NeuralModel.py has implementation of a super-class "NeuralModel" and a bunch of sub-classes extending it like "LSTM", "LSTM_feat", "CNN", "ATTN" and "ATTN_feat"

from methods.neural.nn import *
from tensorflow.contrib.layers import fully_connected

class NeuralModel():
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        self.all_params = all_params
        self.neural_params = all_params['neural_params']
        self.vocab = vocab
        for key in self.neural_params:
            setattr(self, key, self.neural_params[key])
        self.max_length = max_length
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def buildOptimizer(self):
        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.target_cols:
            self.loss[target], self.predict[target], self.accuracy[target] = pred(self.state,
                                                                                  self.n_outputs,
                                                                                  self.weights[target],
                                                                                  self.task_outputs[target])
        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        self.joint_loss = sum(self.loss.values())
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)

    def initialise(self):
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = build_embedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab), self.expand_dims)
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.task_outputs = multi_outputs(self.target_cols)
        self.weights = weight_placeholder(self.target_cols)
        self.keep_prob = tf.placeholder(tf.float32)

    def predict_labels(self, batches, data_batches, weights, savedir):
        return run_pred(self, batches, data_batches, weights, savedir, self.all_params)

    def run_model(self, batches, test_batches, weights):
        return run(self, batches, test_batches, weights, self.all_params)


class LSTM(NeuralModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = False

    def build(self):
        super().initialise()
        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob,self.num_layers,
                                         self.embed, self.sequence_length)
        self.state = state
        super().buildOptimizer()


class LSTM_feat(NeuralModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = True
        self.expand_dims = False

    def build(self):
        super().initialise()
        self.features = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="inputs")
        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob, self.num_layers,
                                         self.embed, self.sequence_length)
        drop_feat = tf.nn.dropout(tf.layers.dense(self.features, self.feature_hidden_size), self.keep_prob)
        drop_rnn = tf.nn.dropout(state, self.keep_prob)
        rnn_feat = tf.reshape(tf.concat([drop_feat, drop_rnn], axis=1), [-1, self.feature_hidden_size + self.hidden_size])
        self.state = rnn_feat
        super().buildOptimizer()


class CNN(NeuralModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = True

    def build(self):
        super().initialise()
        cnn_outputs = cnn(self.embed, self.filter_sizes, self.num_filters, self.keep_prob)
        self.state = cnn_outputs
        super().buildOptimizer()


class ATTN(NeuralModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = False

    def build(self):
        super().initialise()
        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob, self.num_layers,
                                         self.embed, self.sequence_length)
        self.attn = tf.tanh(fully_connected(rnn_outputs, self.attention_size))
        self.alphas = tf.nn.softmax(tf.layers.dense(self.attn, 1, use_bias=False))
        word_attn = tf.reduce_sum(rnn_outputs * self.alphas, 1)
        drop = tf.nn.dropout(word_attn, self.keep_prob)
        self.state = drop
        super().buildOptimizer()


class ATTN_feat(NeuralModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = True
        self.expand_dims = False

    def build(self):
        super().initialise()
        self.features = tf.placeholder(tf.float32, shape=[None, None], name="inputs")
        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob, self.num_layers,
                                         self.embed, self.sequence_length)
        self.attn = tf.tanh(fully_connected(rnn_outputs, self.attention_size))
        self.alphas = tf.nn.softmax(tf.layers.dense(self.attn, 1, use_bias=False))
        word_attn = tf.reduce_sum(rnn_outputs * self.alphas, 1)
        attention = tf.nn.dropout(word_attn, self.keep_prob)
        drop_feat = tf.nn.dropout(self.features, self.keep_prob)
        attn_feat = tf.reshape(tf.concat([drop_feat, attention], axis=1), [-1, self.feature_size + (2 * self.hidden_size)])
        self.state = attn_feat
        super().buildOptimizer()
