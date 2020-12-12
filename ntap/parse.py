import abc
import re
import string
import pandas as pd
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords

class TextPreprocessor:

    op_strs = {'all': ['hashtags', 'mentions', 'links', 'punc', 
                       'digits', 'stopwords', 'stem', 'contractions', 
                       'lowercase', 'ngrams'],
               'clean': ['hashtags', 'mentions', 'links', 'punc', 'digits'],
               'transform': ['stem', 'lowercase', 'ngrams', 'contractions', 'stopwords'],
               'hashtags': ['hashtags'],
               'mentions': ['mentions'], 
               'links': ['links'], 
               'digits': ['digits'],
               'dates': ['dates'],
               'stem': ['stem'],
               'punc': ['punc'],
               'ngrams': ['ngrams'],
               'contractions': ['contractions'],
               'stopwords': ['stopwords'],
               'lowercase': ['lowercase']}


    def __init__(self, f='clean'):

        self.formula = self.__parse_formula(f)
        self.formula.append('shrink_whitespace')


    def transform(self, X, update_formula=None):
        """ Apply stored cleaning and transformation operations to data """
        if update_formula is not None:
            self.formula = self.__parse_formula(formula)

        ops = list()
        for op_str in self.formula:
            ops.append(PreprocessOP(op_str))
        ops.sort()

        if isinstance(X, pd.Series):
            for op in ops:
                if op.name == 'ngrams':
                    pm = self.__get_phrase_model(X)
                    X = X.apply(lambda x: op.func(x, pm=pm))
                else:
                    X = X.apply(op.func)
        elif isinstance(X, list):
            for op in ops:
                if op.name == 'ngrams':
                    pm = self.__get_phrase_model(X)
                    X = [op.func(x, pm=pm) for x in X]
                else:
                    X = [op.func(x) for x in X]
        elif isinstance(X, str):
            for op in ops:
                if op.name == 'ngrams':
                    pass
                else:
                    X = op.func(X)
        return X

    def __parse_formula(self, formula_str):
        if not hasattr(self, "formula"):
            self.formula = list()
        if formula_str.strip() == '':  # is empty
            return self.formula
        else:
            idx = 0
            sign = '+'
            buffer = ""
            neg_ops = list()
            add_ops = list()
            for idx in range(len(formula_str)):
                char = formula_str[idx]
                if char == '+' or char == '-':  # terminate
                    if sign == '+':
                        add_ops.append(buffer)
                    if sign == '-':
                        neg_ops.append(buffer)
                    sign = char
                    buffer = ''
                else:
                    buffer += char
            if len(buffer) > 0:
                if sign == '+':
                    add_ops.append(buffer)
                else:
                    neg_ops.append(buffer)

            for t in add_ops:
                self.formula.extend(self.op_strs[t])
            self.formula = list(set(self.formula))
            for t in neg_ops:
                self.formula = list(set(self.formula) - set(self.op_strs[t]))

            return self.formula

    def __get_phrase_model(self, corpus):
        if isinstance(corpus, pd.Series):
            corpus = corpus.values.tolist()
        corpus = [doc.split() for doc in corpus]
        phrase_model = Phrases(corpus)
        return phrase_model


punc_strs = string.punctuation + "\\\\"
#punc_strs = punc_strs.replace("-", "")
#punc_strs = punc_strs.replace("'", "")

class PreprocessOP:

    trans_fns = {'lowercase': lambda x: x.lower(), 
                 'contractions': lambda x: x.replace('\'', ''),
                 'stopwords': remove_stopwords}
    clean_patterns =  {'links': re.compile(r"(:?http(s)?|pic\.)[^\s]+"),
                       'punc': re.compile(r'[{}]'.format(punc_strs)),
                       'hashtags': re.compile(r'\B#[a-zA-Z0-9_]+'),
                       'mentions': re.compile(r"\B[@]+[a-zA-Z0-9_]+"),
                       'digits': re.compile(r'(?:\$)?(?:\d+|[0-9\-\']{2,})')}


    op_order = ['links', 'hashtags', 'mentions', 
                'digits', 'contractions', 'punc', 'lowercase', 
                'stem', 'ngrams', 'stopwords', 'shrink_whitespace']

    _WHITESPACE_RE = re.compile(r'[\s]+')

    def __init__(self, op_name, lang='english'):

        self.order = self.op_order.index(op_name)
        self.name = op_name
        #stemmer = SnowballStemmer(lang)
        self.trans_fns['stem'] = lambda x: x  # TODO
        self.trans_fns['ngrams'] = lambda x, pm: " ".join(pm[x.split()])

        self.trans_fns['shrink_whitespace'] = lambda x: self._WHITESPACE_RE.sub(' ', x).strip()

        if op_name not in self.clean_patterns and op_name not in self.trans_fns:
            raise ValueError("{} not in list of defined operations".format(op_name))
        self.pattern = op_name
        self.func = op_name

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, key):
        try:
            self._pattern = self.clean_patterns[key]
        except KeyError:
            self._pattern = None

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, op_name):
        if self.pattern is not None:
            self._func = lambda x: self.pattern.sub('', x)
        else:
            self._func = self.trans_fns[op_name]

    def __lt__(self, other):
        return self.order < other.order
