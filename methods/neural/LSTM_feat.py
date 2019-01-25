from tensorflow.contrib.layers import fully_connected
from methods.neural.nn import  *

class ATTN():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build(self):
        self.features = tf.placeholder(tf.int32, shape=[None, None], name="inputs")

        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = build_embedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab))
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.task_outputs = multi_outputs(self.target_cols)

        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.weights = weight_placeholder(self.target_cols)

        self.keep_prob = tf.placeholder(tf.float32)
        self.max_len = tf.placeholder(tf.int32)

        self.network = multi_GRU(self.cell, self.hidden_size, self.keep_prob, self.num_layers)

        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.network, self.embed, self.sequence_length)

        drop_feat = tf.nn.dropout(fully_connected(self.features, self.hidden_size / 2), self.keep_prob)
        drop_rnn = tf.nn.dropout(fully_connected(state, self.hidden_size / 2), self.keep_prob)

        attn_feat = tf.add(drop_feat, drop_rnn)

        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.target_cols:
            self.loss[target], self.predict[target], self.accuracy[target] = pred(attn_feat,
                                                                                          self.n_outputs,
                                                                                          self.weights[target],
                                                                                          self.task_outputs[target])
        self.joint_accuracy = sum(self.accuracy.values()) / len(self.target_cols)
        self.joint_loss = sum(self.loss.values())
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)

    def run_model(self, batches, test_batches, weights):
        return run(self, batches, test_batches, weights)