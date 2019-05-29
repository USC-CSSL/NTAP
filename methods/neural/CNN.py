from methods.neural.baseModel import baseModel

class CNN(baseModel):
    def __init__(self, all_params, max_length, vocab, my_embeddings=None):
        super().__init__(all_params, max_length, vocab, my_embeddings)
        self.feature = False
        self.expand_dims = True

    def build(self):
        self.initialise()
        cnnOutputs = self.build_CNN(self.embed, self.filter_sizes, self.num_filters, self.keep_prob)
        self.state = cnnOutputs
        self.buildOptimizer()
