
"""
from feature_models import TFIDF, LDA, EmbedAvg, DDR

class FeatureSet:

    model_dict = {'tfidf': TFIDF,
                  'lda': LDA,
                  'embed_avg': EmbedAvg,
                  'ddr': DDR,
                  }

    def __init__(self, text_rep_descriptors):
        """ text_rep_descriptors: list of features in form (tfidf|text_col)"""

        self.feature_model_set = dict()
        for desc in text_rep_descriptors:
            if not desc.startswith('(') or not desc.endswith(')') or '|' not in desc:
                raise ValueError("Bad formula item for text representation: {}".format(desc))
            desc = desc.strip('(').strip(')')
            items = desc.split('|')
            if len(items) == 2:
                feature_model_str, text_source = items
            elif len(items) == 3:
                feature_model_str, feature_option, text_source = items
            else:
                raise ValueError("Bad formula RHS ({}): bad |s".format(desc))
            feature_model = self.model_dict[feature_model_str]

            if feature_model_str == 'ddr':
                opts = {'dic': feature_option}
                # kind of clunky. Hard to work around the dictionary input
                feature_model = feature_model(**opts)

            self.feature_model_set[feature_model_str] = feature_model

"""

class NeuralModel:
    def __init__(self, optimizer='adam', learning_rate=0.001, dropout=0.25, hidden_size=64):

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        """
        if self.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer specified")
        """

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

