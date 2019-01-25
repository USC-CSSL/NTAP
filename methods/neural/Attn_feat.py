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
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.features = tf.placeholder(tf.int32, shape=[None, None], name="inputs")

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

        # shape: [batch_size, max_len, attention_size]
        hiddens = tf.tile(tf.reshape(fully_connected(state, self.attention_size), [-1, 1, self.attention_size]),
                          [1, self.max_len, 1])

        # shape: [batch_size, max_len, attention_size]
        summary = fully_connected(rnn_outputs, self.attention_size)

        # sigmoid function on the linear transfer of the sum of hiddens, summary, first and second
        # the production is the vector of attentions, a value between 0 and 1 is assigned to each word
        # shape: [batch_size, max_len, 1]
        attention = tf.reshape(
            fully_connected(tf.add(hiddens, summary), 1, activation_fn=tf.sigmoid), [-1, self.max_len, 1])

        # weighted sum of the hidden states, considering the attention values
        #attentioned_states = tf.reduce_sum(attention * rnn_outputs, axis=1)

        drop_feat = tf.nn.dropout(fully_connected(self.features, self.hidden_size), self.keep_prob)
        drop_attn = tf.nn.dropout(fully_connected(attention * rnn_outputs), self.keep_prob)

        attn_feat = tf.add(drop_feat, drop_attn)

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