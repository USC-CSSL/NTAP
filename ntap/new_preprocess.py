import abc
import re
import string
import pandas as pd
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords
from data import DocTerm, LDA, DDR
from feature_based import TextRegressor

class TextPreprocessor:

    op_strs = {'all': ['hashtags', 'mentions', 'links', 'punc', 
                       'dates', 'digits', 'stopwords', 'stem', 'lowercase', 'ngrams'],
               'clean': ['hashtags', 'mentions', 'links', 'punc', 'dates', 'digits'],
               'transform': ['stem', 'lowercase', 'ngrams', 'stopwords'],
               'hashtags': ['hashtags'],
               'mentions': ['mentions'], 
               'links': ['links'], 
               'numbers': ['dates', 'digits', 'currency'],
               'digits': ['digits'],
               'dates': ['dates'],
               'currency': ['currency'],
               'stem': ['stem'],
               'ngrams': ['ngrams'],
               'lowercase': ['lowercase']}

    def __init__(self, f='clean'):

        self.formula = self.__parse_formula(f)

    @property
    def formula(self):
        return self._formula
    @formula.setter
    def formula(self, f_dict):
        self._formula = f_dict

    def transform(self, X, update_formula=None):
        """ Apply stored cleaning and transformation operations to data """
        if update_formula is not None:
            self.formula = self.__parse_formula(formula)

        ops = list()
        for op_str in self.formula:
            if 'ngrams' == op_str:
                phraser = self.__get_phrase_model(X)
                ops.append(PreprocessOP(op_str, phrase_model=phraser))
            else:
                ops.append(PreprocessOP(op_str))

        if isinstance(X, pd.Series):
            for op in ops:
                X = X.apply(op.func)
        elif isinstance(data, list):
            for op in ops:
                X = [op.func(x) for x in X]
        elif isinstance(data, str):
            for op in ops:
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
        #tqdm.pandas(desc="Bigrams")
        #bigrammed_docs = tokenized_docs.progress_apply(lambda tokens_: phraser[tokens_])

        phrase_model = Phrases(corpus)
        return phrase_model


punc_strs = string.punctuation
punc_strs = punc_strs.replace("-", "") # don't remove hyphens

class PreprocessOP:

    trans_fns = {'lowercase': lambda x: x.lower(), 
                 'stopwords': remove_stopwords} #contractions, POS
    clean_patterns =  {'links': re.compile(r"(http(s)?[^\s]*)|(pic\.[s]*)"),
                       'punc': re.compile(r'[{}]'.format(punc_strs)),
                       'hashtags': re.compile(r"#[a-zA-Z0-9_]+"),
                       'mentions': re.compile(r"@[a-zA-Z0-9_]+"),
                       'dates': re.compile("^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$"),
                       'digits': re.compile(r'(?:\$)?(?:\d+|[0-9\-\']{2,})')}


    def __init__(self, op_name, lang='english', phrase_model=None):

        #stemmer = SnowballStemmer(lang)
        self.trans_fns['stem'] = lambda x: x  # TODO
        if op_name == 'ngrams':
            if phrase_model is None:
                raise ValueError("Whoops!")
            self.trans_fns['ngrams'] = lambda x: " ".join(phrase_model[x.split()])

        if op_name not in self.clean_patterns and op_name not in self.trans_fns:
            raise ValueError("{} not in list of defined operations".format(op_name))
        self.op_type = 'clean' if op_name in self.clean_patterns else 'trans'
        self.pattern = op_name
        self.func = op_name


    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, key):
        if self.op_type == 'clean':
            self._pattern = self.clean_patterns[key]
        else:
            self._pattern = None

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, op_name):
        if self.op_type == 'clean':
            self._func = lambda x: self.pattern.sub('', x)
        else:
            self._func = self.trans_fns[op_name]


if __name__ == '__main__':

    df = pd.read_csv("~/PycharmProjects/HateAnnotations/ghc_with_users.tsv", '\t')
    df.body = TextPreprocessor('all-ngrams').transform(df.body)
    baseline = TextRegressor(formula='Hate~betrayal+(tfidf|body)', data=df)
    print(baseline.single_fit.coef_)
    #dt = DocTerm(df.body)
    #print(dt)
    #lda_m = LDA(df.body, method='mallet', mallet_path="~/mallet-2.0.8/bin/mallet")
    #print(lda_m.model)
