import abc
import re
import string
import pandas as pd
from patsy.desc import Term
from patsy import EvalFactor, EvalEnvironment
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords

from ._formula import parse_formula

#punc_strs = string.punctuation + "\\\\"
#punc_strs = punc_strs.replace('\'', '')
#punc_strs = punc_strs.replace('-', '')


# regexes for cleaning
_LINKS_RE = re.compile(r"(:?http(s)?|pic\.)[^\s]+")
#_PUNC_RE  = re.compile(r'[{}]'.format(punc_strs))
_HASHTAG_RE = re.compile(r'\B#[a-zA-Z0-9_]+')
_MENTIONS_RE = re.compile(r"\B[@]+[a-zA-Z0-9_]+")
_DIGITS_RE = re.compile(r'(?:\$)?(?:\d+|[0-9\-\']{2,})')


# regexes for tokenization
_WORD_RE = re.compile(r"(?:\w|['-]\w)+")
_WORD_NOPUNC_RE = re.compile(r"[\w]+")
_WHITESPACE_RE = re.compile(r'[^\s]+')

whitespace = lambda text: _WHITESPACE_RE.findall(text)

class Clean:

    """ Namespace for text cleaning functions """

    @staticmethod
    def links(text):
        """ Removes hyperlinks (starting with www, http) """
        return _LINKS_RE.sub('', text)

    #@staticmethod
    #def punc(text):
        #""" Removes all punctuation """
        #return _PUNC_RE.sub('', text)

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

    @staticmethod
    def shrink_whitespace(text):
        return " ".join(text.split())

    #ops.sort()


class Tokenize:

    @staticmethod
    def words_nopunc(text):
        return _WORD_NOPUNC_RE.findall(text)

    @staticmethod
    def words(text):
        return _WORD_RE.findall(text)

    @staticmethod
    def whitespace(text):
        return _WHITESPACE_RE.findall(text)

    @staticmethod
    def collocations(tokens, phrase_model):

        #self.fit_collocations(tokenized_data)
        return phrase_model[tokens]

class Preprocessor:

    def __init__(self, formula='hashtags+mentions+links+digits+punc+lowercase'):

        self.tokenize_instr, self.preprocess_instr = parse_formula(formula)

    def transform(self, data):
        """ Applies stored transforms on a list-like object (str)

        Parameters
        ==========
        data : Union[pd.Series, Iterable]
            Either an iterable of strings or a pandas Series object with
            string values. Will apply preprocessing and tokenization to
            each string

        Returns
        =======
        Union[pd.Series, Iterable]
            Return type will match input. Contents will be fully
            transformed following preprocessing and tokenization steps. 

        """

        data_is_pandas = isinstance(data, pd.Series)

        if data_is_pandas:
            updater = lambda fn, l, **kwargs: l.apply(fn, **kwargs)
        else:
            updater = lambda fn, l, **kwargs: [fn(i, **kwargs) for i in l]

        # clean ops
        for term in self.preprocess_instr:
            for e in term.factors:
                eval_str = e.code
                op = getattr(Clean, eval_str)
                data = updater(op, data)

        # fixed clean transformations
        op = getattr(Clean, "shrink_whitespace")
        data = updater(op, data)

        # tokenize ops
        for term in self.tokenize_instr:
            for e in term.factors:
                eval_str = e.code
                op = getattr(Tokenize, eval_str)
                if eval_str == 'collocations':
                    pm = self.fit_collocations(data)
                    data = updater(op, data, phrase_model=pm)
                else:
                    data = updater(op, data)
        return data

    def fit_collocations(self, tokenized_data):
        """ Fit Gensim collocation model. """

        if isinstance(tokenized_data, pd.Series):
            return Phrases(tokenized_data.values.tolist())
        else:
            return Phrases(tokenized_data)

class PreprocessOP:

    funcs = {'contractions': lambda x: x.replace('\'', ''),
             'stopwords': remove_stopwords}

    op_order = ['links', 'hashtags', 'mentions', 
                'digits', 'contractions', 'punc', 'lowercase', 
                'ngrams', 'stopwords', 'shrink_whitespace']

    def __lt__(self, other):
        return self.order < other.order
