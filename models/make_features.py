import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, CategoricalEncoder
from sklearn_pandas import gen_features, CategoricalImputer
from sklearn.impute import SimpleImputer as Imputer

from vectorizers import *
from utils import cosine_similarity, tokenize


# returns transformer list, one per generated/loaded text feature
def get_transformer_list(dataframe,
                         data_dir, 
                         text_col,  # name of the col that contains the document texta
                         methods,  # type of features to load/generate. Can be a name or list of names
                         feature_col= [], # list of columns that are considered as features
                         ordinal_cols = [],
                         categorical_cols = [],
                         bom_method=None,  # options: 'skipgram', 'glove'
                         training_corpus=None,  # options: 'google-news', 'wiki', 'common-crawl'
                         dictionary=None,  # options: 'liwc', 'mfd'
                         comp_measure='cosine-sim',
                         random_seed=0
                         ):
    # Either generates features from text (tfidf, skipgram, etc.) or load from file
    validate_arguments(dataframe, text_col, feature_col, methods)

    transformers = list()

    for method in methods:
        transformation = globals()[method](dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir)
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

    gen_list = ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]

    for method in methods:
        if method not in gen_list:
            print("{} is not an existing method".format(method))
            exit(1)

def tfidf(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir):
    return TfidfVectorizer(min_df=10, stop_words='english',
            tokenizer=tokenize), {'alias': 'tfidf'}

def bagofmeans(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir):
    if training_corpus is None or bom_method is None:
        print("Specify bom_method and training_corpus")
    return (BoMVectorizer(training_corpus,
                         embedding_type=bom_method,
                         tokenizer=tokenize, data_path=data_dir)
                         , {'alias': "_".join([bom_method, training_corpus])})

def ddr(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir):
    if dictionary is None or training_corpus is None or bom_method is None:
        print("Specify dictionary, bom_method, and training_corpus")
        exit(1)
    sim = cosine_similarity if comp_measure == 'cosine-sim' else None
    return (DDRVectorizer(training_corpus,
                         embedding_type=bom_method,
                         tokenizer=tokenize, 
                         data_path=data_dir,
                         dictionary=dictionary,
                         similarity=sim), {'alias': "_".join([bom_method, training_corpus, dictionary])})

def lda(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir):
    num_topics = 100
    return (LDAVectorizer(seed=random_seed,
                         tokenizer=tokenize,
                         num_topics=num_topics),
           {'alias': "LDA_" + str(num_topics) + "topics"})

def dictionary(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir):
    return (DictionaryVectorizer(data_path= data_dir, dictionary_name= dictionary), {"alias": "Dictionary_" + dictionary})