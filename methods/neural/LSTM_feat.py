from methods.neural.baseModel import baseModel
import tensorflow as tf

class LSTM_feat(baseModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = True
        self.expand_dims = False

    def build(self):
        self.initialise()
        self.features = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="inputs")
        rnn_outputs, state = self.dynamic_rnn(self.cell, self.model, self.hidden_layers,
                                         self.keep_prob, self.embed, self.sequence_length)
        drop_feat = tf.nn.dropout(tf.layers.dense(self.features, self.feature_hidden_size), self.keep_prob)
        drop_rnn = tf.nn.dropout(state, self.keep_prob)
        rnn_feat = tf.reshape(tf.concat([drop_feat, drop_rnn], axis=1), [-1, self.feature_hidden_size + self.hidden_layers[-1]])
        self.state = rnn_feat
        self.buildPredictor()
