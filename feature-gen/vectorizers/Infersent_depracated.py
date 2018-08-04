import torch

class InfersentVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, tokenizer=None, glove_file="glove.840B.300d.txt",
                        model_file="infersent.allnli.pickle"):
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "sent_embeddings", "infersent", model_file)
        self.glove_path = os.path.join(data_dir, "word_embeddings", "GloVe", glove_file)
        if not os.path.isfile(self.glove_path):
            print("Couldn't find GloVe file in %s. Exiting" % self.glove_path)
            exit(1)
        if not os.path.isfile(self.model_path):
            print("Couldn't find infersent model file in %s. Exiting" % self.model_path)
            exit(1)
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        self.model = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        if self.tokenizer is None:
            self.sentences = X
            self.model.set_glove_path(self.glove_path, tokenize=True)
        else:
            self.sentences = [" ".join(self.tokenizer(sent)) for sent in X]
            self.model.set_glove_path(self.glove_path)
            self.model.build_vocab(self.sentences)
        return self

    def transform(self, X, y=None):
        return self.model.encode(self.sentences)
