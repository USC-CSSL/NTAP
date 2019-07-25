from methods.neural.baseModel import baseModel

class LSTM(baseModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = False

    def build(self):
        self.initialise()
        rnn_outputs, state = self.dynamic_rnn(self.cell, self.model, self.hidden_layers,
                                         self.keep_prob, self.embed, self.sequence_length)
        self.state = state
        self.buildPredictor()
