from nltk import tokenize as nltk_token
nltk_tokenizer = nltk_token.TreebankWordTokenizer()
nltk_tweettokenize = nltk_token.TweetTokenizer()

from scipy import spatial
from happierfuntokenizing import HappierTokenizer

def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens

def happiertokenize(text):
    tok = HappierTokenizer(preserve_case=False)
    return tok.tokenize(text)

def tweettokenize(text):
    # keeps #s and @s appended to their next word
    return nltk_tweettokenize.tokenize(text)