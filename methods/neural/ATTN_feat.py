from methods.neural.baseModel import baseModel
from tensorflow.contrib.layers import fully_connected

class ATTN_feat(baseModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = True
        self.expand_dims = False

    def build(self):
        self.initialise()
        self.features = tf.placeholder(tf.float32, shape=[None, None], name="inputs")
        rnn_outputs, state = self.dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob, self.num_layers,
                                         self.embed, self.sequence_length)
        self.attn = tf.tanh(fully_connected(rnn_outputs, self.attention_size))
        self.alphas = tf.nn.softmax(tf.layers.dense(self.attn, 1, use_bias=False))
        word_attn = tf.reduce_sum(rnn_outputs * self.alphas, 1)
        attention = tf.nn.dropout(word_attn, self.keep_prob)
        drop_feat = tf.nn.dropout(self.features, self.keep_prob)
        attn_feat = tf.reshape(tf.concat([drop_feat, attention], axis=1), [-1, self.feature_size + (2 * self.hidden_size)])
        self.state = attn_feat
        self.buildOptimizer()
