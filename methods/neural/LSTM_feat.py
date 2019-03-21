from tensorflow.contrib.layers import fully_connected
from methods.neural.nn import  *

class LSTM_feat():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        self.feature = True
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build(self):
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.features = tf.placeholder(tf.float32, shape=[None, None], name="inputs")

        self.embedding_placeholder = build_embedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab))
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.task_outputs = multi_outputs(self.target_cols)

        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.weights = weight_placeholder(self.target_cols)

        self.keep_prob = tf.placeholder(tf.float32)

        #self.network = multi_GRU(self.cell, self.hidden_size, self.keep_prob, self.num_layers)

        rnn_outputs, state = dynamic_rnn(self.cell, self.model, self.hidden_size,
                                         self.keep_prob, self.num_layers,
                                         self.embed, self.sequence_length)


        drop_feat = tf.nn.dropout(self.features, self.keep_prob)
        drop_rnn = tf.nn.dropout(state, self.keep_prob)

        rnn_feat = tf.reshape(tf.concat([drop_feat, drop_rnn], axis=1), [-1, self.feature_size + self.hidden_size])

        self.loss, self.accuracy, self.predict = dict(), dict(), dict()

        for target in self.target_cols:
            print(self.n_outputs)
            self.loss[target], self.predict[target], self.accuracy[target] = pred(rnn_feat,
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