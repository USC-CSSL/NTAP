
from nltk import tokenize as nltk_token

nltk_tokenizer = nltk_token.TreebankWordTokenizer()

treebank_tokenizer = nltk_token.TreebankWordTokenizer()
wordpunc_tokenizer = nltk_token.WordPunctTokenizer()

from happierfuntokenizing import HappierTokenizer
def tokenize(text):

    #tokens = nltk_tokenizer.tokenize(text)
    tokens = wordpunc_tokenizer.tokenize(text)
    return tokens

def happiertokenize(text):
    tok = HappierTokenizer(preserve_case=False)
    return tok.tokenize(text)

def tweettokenize(text):
    # keeps #s and @s appended to their next word
    return nltk_tweettokenize.tokenize(text)
