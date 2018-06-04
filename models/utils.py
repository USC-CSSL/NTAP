

from nltk import tokenize as nltk_token
nltk_tokenizer = nltk_token.TreebankWordTokenizer()
from scipy import spatial

def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens
