import os 
import numpy as np
import json
from sys import stdout
import sys

from sklearn.base import BaseEstimator, TransformerMixin

# Different versions of fasttext 
try:
    import fasttext as fastText
except ModuleNotFoundError:
    import fastText

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, stop_words):
        self.tokenizer = tokenizer
        self.remove_regex = re.compile(r"(?:{})".format("|".join(stop_words)))
        self.model_path = os.environ["FASTTEXT_PATH"]
        if not os.path.isfile(self.model_path):
            print("Couldn't find fasttext .bin file in %s. Exiting" % self.model_path)
            exit(1)

    def fit(self, X, y=None):
        print("Loading fasttext model")
        self.trained_model = fastText.load_model(self.model_path)
        return self

    def transform(self, X, y=None):
        sentences = list()

        print("Encoding sentences with FastText")
        for i, sent in enumerate(X):
            stdout.write("\r{:.2%} done".format(float(i) / len(X)))
            stdout.flush()
            if self.remove_regex is not None:
                sent = self.remove_regex.sub(" ", sent)
            sentences.append(list(self.trained_model.get_sentence_vector(sent)))
        return np.array(sentences, dtype='float32')
