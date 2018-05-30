import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, CategoricalEncoder
from sklearn_pandas import gen_features, CategoricalImputer
from sklearn.impute import SimpleImputer as Imputer

from vectorizers import *

from nltk import tokenize as nltk_token
nltk_tokenizer = nltk_token.TreebankWordTokenizer()
alpha_re = re.compile(r"[^a-zA-Z\s]")
length_re = re.compile(r'\w{3,}')


def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens


# returns transformer list, one per generated/loaded text feature
def get_text_transformer(dataframe,
                         text_col,  # name of the col that contains the document texta
                         methods,  # type of features to load/generate. Can be a name or list of names
                         feature_col= [], # list of columns that are considered as features
                         ordinal_cols = [],
                         categorical_cols = [],
                         bom_method=None,  # options: 'skipgram', 'glove'
                         training_corpus=None,  # options: 'google-news', 'wiki', 'common-crawl'
                         dictionary=None  # options: 'liwc', 'mfd'
                         ):
    # Either generates features from text (tfidf, skipgram, etc.) or load from file
    validate_arguments(dataframe, text_col, feature_col, methods)

    transformers = list()

    for method in methods:
        transformation = globals()[method](dataframe, text_col, bom_method, training_corpus, dictionary)
        transformers.append((text_col, ) + transformation) if type(transformation) == tuple else (text_col, transformation)

    if len(feature_col) > 0:
        transformers += gen_features(
                columns=[ [col] for col in feature_col],
                classes=[StandardScaler])

    if len(categorical_cols) > 0:
        transformers += gen_features(
                    columns=[ [col] for col in categorical_cols],
                    classes=[CategoricalImputer, CategoricalEncoder]
                            )

    if len(ordinal_cols) > 0:
        transformers += gen_features(
                    columns=[ [col] for col in ordinal_cols],
                    classes=[{'class': Imputer, 'missing_values':-1},
                             {'class': MinMaxScaler}
                            ])

    return transformers


def validate_arguments(dataframe, text_col, feature_col, methods):

    # text_col
    if text_col not in dataframe.columns:
        print("Not valid text column")
        exit(1)

    # feature_col
    if type(feature_col) != list:
        print("feature_col should be a list of feature column names")
        exit(1)
    if len(feature_col) > 0:
        if not set(text_col).issubset(dataframe.columns):
            print("To load LIWC/MFD features, load dataframe with \'feature_col\' as columns")
            exit(1)

    # methods
    if type(methods) != list:
        methods = [methods]

    gen_list = ['tfidf', 'bagofmeans', 'ddr', 'fasttext', 'infersent']

    for method in methods:
        if method not in gen_list:
            print("{} is not an existing method".format(method))
            exit(1)




def tfidf(dataframe, text_col, bom_method, training_corpus, dictionary):
    return TfidfVectorizer(min_df=10, stop_words='english',
            tokenizer=tokenize), {'alias': 'tfidf'}


def bagofmeans(dataframe, text_col, bom_method, training_corpus, dictionary):
    if training_corpus is None or bom_method is None:
        print("Specify bom_method and training_corpus")
        exit(1)
    return BoMVectorizer(training_corpus,tokenizer=tokenize)


def ddr(dataframe, text_col, bom_method, training_corpus, dictionary):
    if dictionary is None or training_corpus is None or bom_method is None:
        print("Specify dictionary, bom_method, and training_corpus")
        exit(1)
    return [BoMVectorizer(training_corpus,
                              tokenizer=tokenize),
                DDRVectorizer(dictionary=dictionary,
                              similarity='cosine-sim'),
                StandardScaler
                ]

def lda(dataframe, text_col, bom_method, training_corpus, dictionary):
    return LDAVectorizer(dataframe, text_col)