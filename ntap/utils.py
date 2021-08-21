import re
import requests

from tqdm import tqdm
import pandas as pd
from patsy.desc import Term
from patsy import EvalFactor, EvalEnvironment, ModelDesc
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import remove_stopwords
from stop_words import get_stop_words

def download_file(url, local_dest):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size= 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(local_dest, 'wb') as fo:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            fo.write(data)
    progress_bar.close()
    #if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        #self.log.error(f"Could not download {url} to {local_dest}")

#def load_imdb(file_name):
    #d = {'pos': list(), 'neg': list()}
    #zf = zipfile.ZipFile(file_name, 'r')
    #for name in zf.namelist():
        #if name.startswith('pos') and name.endswith('txt'):
            #data = zf.read(name)
            #d['pos'].append(data.strip())
        #if name.startswith('neg') and name.endswith('txt'):
            #data = zf.read(name)
            #d['neg'].append(data.strip())
    #return d


_VALID_TOKENIZERS = ['words', 'words_nopunc', 'whitespace']
_DEFAULT_TOKENIZER = 'words'

def parse_formula(f_str):

    patsy_formula = ModelDesc.from_formula(f_str)

    tokenize = patsy_formula.lhs_termlist

    valid_tokenizers = list()
    for term in tokenize:
        for e in term.factors:
            code = e.code
            if code in _VALID_TOKENIZERS:
                valid_tokenizers.append(code)

    if len(valid_tokenizers) == 0:
        tokenize.insert(0, Term([EvalFactor(_DEFAULT_TOKENIZER)]))
    if len(valid_tokenizers) > 1:
        raise RuntimeError("Multiple tokenizers found in formula\n"
                           f"Specify one from {' '.join(_VALID_TOKENIZERS)}")

    preprocess = [t for t in patsy_formula.rhs_termlist 
                  if len(t.factors) > 0]
    return tokenize, preprocess


# TODO: add "update" syntax to model objects, with the idea of comparing
# models across different settings (formulas). For example, tfidf versus
# BERT, or stemming versus non-stemming


# regexes for cleaning
_LINKS_RE = re.compile(r"(:?http(s)?|pic\.)[^\s]+")
#_PUNC_RE  = re.compile(r'[{}]'.format(punc_strs))
_HASHTAG_RE = re.compile(r'\B#[a-zA-Z0-9_]+')
_MENTIONS_RE = re.compile(r"\B[@]+[a-zA-Z0-9_]+")
_DIGITS_RE = re.compile(r'(?:\$)?(?:\d+|[0-9\-\']{2,})')


# regexes for tokenization
_WORD_RE = re.compile(r"[a-zA-Z]{2,20}(?:[\-'][a-zA-Z]{1,20})?")
_WORD_NOPUNC_RE = re.compile(r"[a-zA-Z]{2,20}")
_WHITESPACE_RE = re.compile(r'[^\s]+')

class Clean:

    """ Namespace for text cleaning functions """

    @staticmethod
    def social(text):
        """ Remove links, hashtags, and mentions """
        sub1 = _LINKS_RE.sub('', text)
        sub2 = _MENTIONS_RE.sub('', sub1)
        return _HASHTAG_RE.sub('', sub2)


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
    def stopwords(tokens, lang='en'):
        stop_list = set(get_stop_words(lang))
        return [w for w in tokens if w not in stop_list]

    @staticmethod
    def collocations(tokens, phrase_model):

        #self.fit_collocations(tokenized_data)
        return phrase_model[tokens]

class Preprocessor:

    def __init__(self, formula='words~hashtags+mentions+links+lowercase'):

        self.tokenize_instr, self.preprocess_instr = parse_formula(formula)

        readable_formula = " + ".join([term.name() for term in
                                       self.tokenize_instr])
        readable_formula += " ~ "
        readable_formula += " + ".join([term.name() for term in
                                       self.preprocess_instr])
        self.description = readable_formula

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

#class PreprocessOP:

    #op_order = ['links', 'hashtags', 'mentions', 
                #'digits', 'contractions', 'punc', 'lowercase', 
                #'ngrams', 'stopwords', 'shrink_whitespace']

    #def __lt__(self, other):
        #return self.order < other.order
