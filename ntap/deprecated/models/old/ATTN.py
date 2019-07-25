from methods.neural.baseModel import baseModel
from tensorflow.contrib.layers import fully_connected
import tensorflow as tf

class ATTN(baseModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = False

    def build(self):
        self.initialise()
        rnn_outputs, state = self.dynamic_rnn(self.cell, self.model, self.hidden_layers,
                                         self.keep_prob, self.embed, self.sequence_length)
        self.attn = tf.tanh(fully_connected(rnn_outputs, self.attention_size))
        self.alphas = tf.nn.softmax(tf.layers.dense(self.attn, 1, use_bias=False))
        word_attn = tf.reduce_sum(rnn_outputs * self.alphas, 1)
        drop = tf.nn.dropout(word_attn, self.keep_prob)
        self.state = drop
        self.buildPredictor()
