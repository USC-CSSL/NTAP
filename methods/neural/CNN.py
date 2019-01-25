from methods.neural.nn import *


class CNN():
    def __init__(self, params, max_length, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        self.max_length = max_length
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build(self):
        tf.reset_default_graph()
        self.embedding_placeholder = tf.expand_dims(build_embedding(self.pretrain, self.train_embedding,
                                                     self.embedding_size, len(self.vocab)), -1)

        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.weights = weight_placeholder(self.target_cols)
        self.task_outputs = multi_outputs(self.target_cols)

        self.keep_prob = tf.placeholder(tf.float32)

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