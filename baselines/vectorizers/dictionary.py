import re, os
from sys import stdout
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DICT_PATH = os.environ["DICTIONARIES"]

class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dictionary_name):
        self.dictionary_name = dictionary_name
        dictionary_path = os.path.join(DICT_PATH, dictionary_name + '.dic')
        self.dictionary = dict()
        self.feature_names = dict()
        try:
            with open(dictionary_path, 'r') as dic:
                c = 1
                dic_part = False
                for line in dic:
                    line = line.replace("\n", "").lstrip().rstrip()
                    tokens = line.split()
                    if len(tokens) == 0:
                        continue
                    if c == 1:
                        if line != "%":
                            print("Dictionary format incorrect. Expecting % in the first line.")
                        else:
                            dic_part = True
                    elif dic_part:
                        if line == "%":
                            dic_part = False
                        else:
                            self.feature_names[tokens[0]] = tokens[1]
                    else:
                        num_start = 0
                        key = ""
                        for token in tokens:
                            if not token.isdigit():
                                key += " " + token
                                num_start += 1
                        for token in tokens[num_start:]:
                            self.dictionary.setdefault(self.feature_names[token], []).append(key)
                    c += 1

        except FileNotFoundError:
            print("Could not load dictionary %s from %s" % (self.dictionary_name, dictionary_path))
            exit(1)

    def fit(self, X, y=None):
        self.dictionary_re = dict()
        for cat, words in self.dictionary.items():
            self.dictionary_re[cat] = list()
            for word in words:
                word = word.replace(")", "\\)").replace("(", "\\(").replace(":", "\\:").replace(";", "\\;").replace("/", "\\/")
                if len(word) == 0:
                    continue
                if word[-1] == "*":
                    self.dictionary_re[cat].append(re.compile("(\\b" + word[:-1] + "+\w*\\b)"))
                else:
                    self.dictionary_re[cat].append(re.compile("(\\b" + word + "\\b)"))
        return self

    def transform(self, X, y=None):
        vectors = list()
        c = 0
        for sentence in X:
            c += 1
            stdout.write("\r{:.2%} done".format(float(c) / len(X)))
            stdout.flush()
            vectors.append(self.count_sentence(sentence))
        return np.array(vectors)


    def count_sentence(self, sentence):
        vector = []
        for cat in sorted(self.dictionary_re.keys()):
            count = 0
            for reg in self.dictionary_re[cat]:
                x = len(re.findall(reg, sentence))
                if x > 0:
                    count += x
            if len(sentence.split()) == 0:
                vector.append(0)
            else:
                vector.append(float(count) / float(len(sentence.split())))
        return vector

    def get_feature_names(self):
        return sorted(self.dictionary)
