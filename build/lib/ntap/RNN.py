import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
from ntap.models import Model

class RNN(Model):
    """Recurrent Neural Network class.

    RNN class that inherits from Model. Provides model building,
    training, evaluating, and predicting for RNN models.


    Attributes:
           optimizer: An optimizer to be used during training.
           embedding_source: A path to embeddings.
           hidden_size: The number of hidden units in the RNN layer(s).
           bi: A bool indicating whether or not the cells are bidirectional.
           cell_type: A string indicating the type of RNN cell.
           rnn_dropout: The dropout rate for the RNN layer.
           embedding_dropout: The dropout rate for embeddings.
           rnn_pooling: A string indicating type of pooling.
           random_state: An optional int, setting the random state.
           learning_rate: A float indicating the learning rate for the model.
           vars: A dictionary to store all the network model variables.
           init:
    """
    def __init__(self, formula, data, hidden_size=128, cell='biLSTM',
            rnn_dropout=0.5, embedding_dropout=None, optimizer='adam',
            learning_rate=0.001, rnn_pooling='last',
            embedding_source='glove', random_state=None):
        Model.__init__(self, optimizer=optimizer,
                embedding_source=embedding_source)

        self.hidden_size = hidden_size
        self.bi = cell.startswith('bi')
        self.cell_type = cell[2:] if self.bi else cell
        self.rnn_dropout = rnn_dropout
        self.embedding_dropout = embedding_dropout
        #self.max_seq = data.max_len  # load from data OBJ
        self.rnn_pooling = rnn_pooling
        self.random_state = random_state
        self.learning_rate = learning_rate

        self.vars = dict() # store all network variables
        self.__parse_formula(formula, data)

        self.build(data)

    def set_params(self, **kwargs):
        print("TODO")

    def __parse_formula(self, formula, data):
        lhs, rhs = [s.split("+") for s in formula.split('~')]
        for target in lhs:
            target = target.strip()
            if target in data.targets:
                print("Target already present: {}".format(target))
            elif target in data.data.columns:
                data.encode_targets(target, encoding='labels')  # sparse
            else:
                raise ValueError("Failed to load {}".format(target))
        for source in rhs:
            # can't have two of (seq, bag,...)
            source = source.strip()
            if source.startswith("seq("):
                # get sequence of int id inputs
                text_col = source.replace("seq(", "").replace(")", "")
                data.encode_docs(text_col)
                if not hasattr(data, "embedding"):
                    data.load_embedding(text_col)
                # data stored in data.inputs[text_col]
            elif source.startswith("bag("):
                # multi-instance learning!
                # how to aggregate? If no param set, do rebagging with default size
                print("TODO")
            elif source in data.features:
                inputs.append(source)
            elif source == 'tfidf':
                print("Fetch tfidf from features")
            elif source == 'lda':
                print("Fetch lda from features")
            elif source == 'ddr':
                print("Write DDR method")
            elif source.startswith('tfidf('):
                text_col = source.replace('tfidf(','').strip(')')
                if text_col not in data.data.columns:
                    raise ValueError("Could not parse {}".format(source))
                    continue
                data.tfidf(text_col)
            elif source.startswith('lda('):
                text_col = source.replace('lda(','').strip(')')
                if text_col not in data.data.columns:
                    raise ValueError("Could not parse {}".format(source))
                    continue
                data.lda(text_col)
            elif source in data.data.columns:
                data.encode_inputs(source)
            else:
                raise ValueError("Could not parse {}".format(source))

    def build(self, data):
        tf.reset_default_graph()
        self.vars["sequence_length"] = tf.placeholder(tf.int32, shape=[None],
                name="SequenceLength")
        self.vars["word_inputs"] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="RNNInput")
        self.vars["keep_ratio"] = tf.placeholder(tf.float32, name="KeepRatio")
        W = tf.Variable(tf.constant(0.0, shape=[len(data.vocab), data.embed_dim]), trainable=False, name="Embed")
        self.vars["Embedding"] = tf.nn.dropout(tf.nn.embedding_lookup(W,
                self.vars["word_inputs"]), rate=self.vars["keep_ratio"], name="EmbDropout")
        self.vars["EmbeddingPlaceholder"] = tf.placeholder(tf.float32,
                shape=[len(data.vocab), data.embed_dim])
        self.vars["EmbeddingInit"] = W.assign(self.vars["EmbeddingPlaceholder"])
        self.vars["states"] = self.__build_rnn(self.vars["Embedding"],
                self.hidden_size, self.cell_type, self.bi,
                self.vars["sequence_length"])

        if self.rnn_dropout is not None:
            self.vars["hidden_states"] = tf.nn.dropout(self.vars["states"],
                                                           rate=self.vars["keep_ratio"],
                                                           name="RNNDropout")
        else:
            self.vars["hidden_states"] = self.vars["states"]

        for target in data.targets:
            n_outputs = len(data.target_names[target])
            self.vars["target-{}".format(target)] = tf.placeholder(tf.int64,
                    shape=[None], name="target-{}".format(target))
            self.vars["weights-{}".format(target)] = tf.placeholder(tf.float32,
                    shape=[n_outputs], name="weights-{}".format(target))
            logits = tf.layers.dense(self.vars["hidden_states"], n_outputs)
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.vars["target-{}".format(target)])
            xentropy = tf.losses.sparse_softmax_cross_entropy\
                (labels=self.vars["target-{}".format(target)],
                    logits=logits, weights=weight)
            self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
            self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)
            self.vars["accuracy-{}".format(target)] = tf.reduce_mean(
                tf.cast(tf.equal(self.vars["prediction-{}".format(target)],
                                 self.vars["target-{}".format(target)]), tf.float32))

        self.vars["joint_loss"] = sum([self.vars[name] for name in self.vars if name.startswith("loss")])
        self.vars["joint_accuracy"] = sum([self.vars[name] for name in self.vars if name.startswith("accuracy")]) \
                                      / len([self.vars[name] for name in self.vars if name.startswith("accuracy")])
        if self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer specified")
        self.vars["training_op"] = opt.minimize(loss=self.vars["joint_loss"])
        self.init = tf.global_variables_initializer()


    def list_model_vars(self):
        # return list of variable names that can be retrieved during inference
        vs = [v for v in self.vars if v.startswith("loss-")]
        vs.append("hidden_states")
        if isinstance(self.rnn_pooling, int):
            vs.append("rnn_alphas")
        return vs

    def __build_rnn(self, inputs, hidden_size, cell_type, bi, sequences, peephole=False):
        if cell_type == 'LSTM':
            if bi:
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="ForwardRNNCell",
                          state_is_tuple=False)
                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="BackwardRNNCell",
                          state_is_tuple=False)
            else:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="RNNCell",
                          dtype=tf.float32)
        elif cell_type == 'GRU':
            if bi:
                fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          name="ForwardRNNCell")
                bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          reuse=False, name="BackwardRNNCell")
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          name="BackwardRNNCell", dtype=tf.float32)
        if bi:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                inputs, dtype=tf.float32, sequence_length=sequences)
            hidden_states = tf.concat(outputs, 2)  # shape (B, T, 2*h)
            state = tf.concat(states, 1)  # last unit
        else:
            hidden_states, state = tf.nn.dynamic_rnn(cell, inputs,
                    dtype=tf.float32, sequence_length=sequences)

        if isinstance(self.rnn_pooling, int):
            return self.__attention(hidden_states, self.rnn_pooling)
        elif self.rnn_pooling == 'last':  # default
            return state
        elif self.rnn_pooling == 'max':
            return tf.reduce_max(hidden_states, reduction_indices=[1])
        elif self.rnn_pooling == 'mean':
            return tf.reduce_mean(hidden_states, axis=1)

    def __attention(self, inputs, att_size):
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, att_size],
            stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))

        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        self.vars["rnn_alphas"] = alphas
        return output