from methods.neural.nn import *
from tensorflow.python.ops import array_ops

class RCNN():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build(self):
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = build_embedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab))
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.task_outputs = multi_outputs(self.target_cols)

        self.sequence_length = tf.placeholder(tf.int32, [None])

        self.weights = weight_placeholder(self.target_cols)
        self.keep_prob = tf.placeholder(tf.float32)
        self.max_len = tf.placeholder(tf.float32)

        forward_network = multi_GRU(self.cell, self.hidden_size, self.keep_prob, self.num_layers)
        backward_network = multi_GRU(self.cell, self.hidden_size, self.keep_prob, self.num_layers)

        self.forward_first = tf.zeros((tf.shape(self.embed)[0], 1, self.embedding_size))
        self.backward_first = tf.zeros((tf.shape(self.embed)[0], 1, self.embedding_size))

        self.forward_rnn_outputs, _ = tf.nn.dynamic_rnn(forward_network, tf.concat([self.forward_first, self.embed], 1), dtype=tf.float32, sequence_length=self.sequence_length)

        embed_rev = array_ops.reverse_sequence(self.embed, seq_lengths=self.sequence_length, seq_dim=1)
        backward_rnn_temp, _ = tf.nn.dynamic_rnn(backward_network, tf.concat([self.backward_first, embed_rev], 1), dtype=tf.float32, sequence_length=self.sequence_length)

        self.backward_rnn_outputs = array_ops.reverse_sequence(backward_rnn_temp, seq_lengths=self.sequence_length, seq_dim=1)

        self.forward_embed_backward = tf.expand_dims(tf.concat([tf.slice(self.forward_rnn_outputs, [0, 0, 0], [tf.shape(self.embed)[0], tf.shape(self.embed)[1], self.hidden_size]), self.embed, tf.slice(self.backward_rnn_outputs, [0, 0, 0], [tf.shape(self.embed)[0], tf.shape(self.embed)[1], self.hidden_size])], 2), -1)

        self.cnn_outputs = cnn(self.forward_embed_backward, self.filter_sizes, self.num_filters, self.keep_prob)

        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.target_cols:
            self.loss[target], self.predict[target], self.accuracy[target] = pred(self.cnn_outputs,
                                                                                  self.n_outputs,
                                                                                  self.weights[target],
                                                                                  self.task_outputs[target])
        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        self.joint_loss = sum(self.loss.values())
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)



    def run_model(self, batches, test_batches, weights):
        return run(self, batches, test_batches, weights)

    def predict_labels(self, batches, data_batches, weights, savedir):
        return run_pred(self, batches, data_batches, weights, savedir)