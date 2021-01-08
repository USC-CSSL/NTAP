import abc
import re
import string
import pandas as pd
from patsy.desc import Term
from patsy import EvalFactor, EvalEnvironment
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords

from ._formula import parse_formula

punc_strs = string.punctuation + "\\\\"


# regexes for cleaning
_LINKS_RE = re.compile(r"(:?http(s)?|pic\.)[^\s]+")
_PUNC_RE  = re.compile(r'[{}]'.format(punc_strs))
_HASHTAG_RE = re.compile(r'\B#[a-zA-Z0-9_]+')
_MENTIONS_RE = re.compile(r"\B[@]+[a-zA-Z0-9_]+")
_DIGITS_RE = re.compile(r'(?:\$)?(?:\d+|[0-9\-\']{2,})')


# regexes for tokenization
_WORD_RE = re.compile(r'[\w]{2,20}')
_WHITESPACE_RE = re.compile(r'[^\s]+')

word = lambda text: _WORD_RE.findall(text)
whitespace = lambda text: _WHITESPACE_RE.findall(text)
shrink_whitespace = lambda text: " ".join(text.split())

#if name not in self.tokenizers:
    #raise ValueError(f"Tokenizer {name} not found. Options are: "
                     #f"{', '.join(list(self.tokenizers.keys()))}")

def transform(self, docs):
    """ Return list of tokens for each doc in docs 

    Parameters
    ----------
    docs : list-like
        Iterable over string documents. Can be list, 
        numpy ndarray, or pandas Series

    Returns
    -------
    list-like (list, array, Series)
        Iterable of token-lists. Type will match input

    """

    if isinstance(docs, pd.Series):
        return docs.apply(self.tokenizer)
    elif isinstance(docs, list):
        return [self.tokenizer(doc) for doc in docs]
    elif isinstance(docs, np.ndarray):
        return np.array([self.tokenizer(doc) for doc in docs])
    else:
        raise RunTimeError("Type of input not recognized. Must be "
                           "list, ndarray, or Series")


class Clean:

    """ Namespace for text cleaning functions """

    @staticmethod
    def links(text):
        """ Removes hyperlinks (starting with www, http) """
        return _LINKS_RE.sub('', text)

    @staticmethod
    def punc(text):
        """ Removes all punctuation """
        return _PUNC_RE.sub('', text)

    @staticmethod
    def hashtags(text):
        """ Removes tokens starting with \"#\" """
        return _HASHTAG_RE.sub('', text)

    @staticmethod
    def mentions(text):
        """ Removes tokens starting with at least one \"@\" """
        return _MENTIONS_RE.sub('', text)

    @staticmethod
    def digits(text):
        """ Removes numeric digits, including currency and dates """
        return _DIGITS_RE.sub('', text)

    @staticmethod
    def lowercase(text):
        return text.lower()

    def ngrams(self, text):
        #ops.sort()

        if not hasattr("collocations"):
            self.fit_collocations(data)

class Preprocessor:

    def __init__(self, formula='hashtags+mentions+links+digits+punc+lowercase'):

        self.tokenize_instr, self.preprocess_instr = parse_formula(formula)

        print(self.tokenize_instr)
        print(self.preprocess_instr)
        #self.__build_transformer()

    def transform(self, data):

        print(data)
        data_is_pandas = isinstance(data, pd.Series)

        if data_is_pandas:
            updater = lambda fn, l: l.apply(fn) 
        else:
            updater = lambda fn, l: [fn(i) for i in l]

        funcs = list()
        for term in self.preprocess_instr:
            for e in term.factors:

                eval_str = e.code
                op = getattr(Clean, eval_str)
                data = updater(op, data)
                #state = {}
                #eval_env = EvalEnvironment.capture(0)
                #passes = e.memorize_passes_needed(state, eval_env)
                #returned = eval_env.eval(e.code, inner_namespace=vars(Clean))
                #mat = e.eval(state, data)
                #print(returned)

        print(data)
        exit(1)

        if isinstance(X, pd.Series):
            for op in ops:
                if op.name == 'ngrams':
                    pm = self.__get_phrase_model(X)
                    X = X.apply(lambda x: op.func(x, pm=pm))
                else:
                    X = X.apply(op.func)
        elif isinstance(X, (np.ndarray, list)):
            for op in ops:
                if op.name == 'ngrams':
                    pm = self.__get_phrase_model(X)
                    X = [op.func(x, pm=pm) for x in X]
                else:
                    X = [op.func(x) for x in X]
        return X

    def fit_collocations(self, data):
        """ Fit Gensim collocation model. 

        TODO: Remove gensim dependency, add options 

        """

        if isinstance(data, pd.Series):
            corpus = corpus.values.tolist()
        else:
            corpus = data
        corpus = [doc.split() for doc in corpus]
        self.phrase_model = Phrases(corpus)

class PreprocessOP:

    funcs = {'lowercase': lambda x: x.lower(), 
             'contractions': lambda x: x.replace('\'', ''),
             'stopwords': remove_stopwords}


    op_order = ['links', 'hashtags', 'mentions', 
                'digits', 'contractions', 'punc', 'lowercase', 
                'ngrams', 'stopwords', 'shrink_whitespace']


    def __init__(self, op_name, lang='english'):

        self.order = self.op_order.index(op_name)
        self.name = op_name
        self.trans_fns['ngrams'] = lambda x, pm: " ".join(pm[x.split()])

        self.trans_fns['shrink_whitespace'] = lambda x: self._WHITESPACE_RE.sub(' ', x).strip()

        if op_name not in self.clean_patterns and op_name not in self.trans_fns:
            raise ValueError("{} not in list of defined operations".format(op_name))
        self.pattern = op_name
        self.func = op_name

    def __lt__(self, other):
        return self.order < other.order
