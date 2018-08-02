import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import gen_features, CategoricalImputer, DataFrameMapper
from scipy import sparse

from vectorizers import *
from utils import cosine_similarity, tokenize, happiertokenize


# returns transformer list, one per generated/loaded text feature
def get_transformer_list(dataframe,
                         data_dir, 
                         text_col,  # name of the col that contains the document texta
                         methods,  # type of features to load/generate. Can be a name or list of names
                         feature_cols= [], # list of columns that are considered as features
                         ordinal_cols = [],
                         categorical_cols = [],
                         ngrams = [],
                         bom_method=None,  # options: 'skipgram', 'glove'
                         training_corpus=None,  # options: 'google-news', 'wiki', 'common-crawl'
                         dictionary=None,  # options: 'liwc', 'mfd'
                         comp_measure='cosine-sim',
                         random_seed=0,
                         feature_reduce=None,
                         tokenizer= "tokenize"
                         ):
    # Either generates features from text (tfidf, skipgram, etc.) or load from file

    sent_tokenizer = tokenize if tokenizer == "tokenize" else happiertokenize

    transformers = list()

    for method in methods:
        transformation = globals()[method](dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams, sent_tokenizer,"cosine-sim")
        transformers.append((text_col, ) + transformation) if type(transformation) == tuple else (text_col, transformation)

    if len(feature_cols) > 0:
        transformers += gen_features(
                columns=[ [col] for col in feature_cols])
                #classes=[StandardScaler])
    """
    if len(categorical_cols) > 0:
        print("FUCKINGNNGKNDKNFKD")
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

    """

    mapper = DataFrameMapper(transformers, sparse=True, input_df=True)
    X = mapper.fit_transform(dataframe)
    lookup_dict = {i: feat for i, feat in enumerate(mapper.transformed_names_)}

    # as in DLATK, rare features that occur for less that <feature_reduce> percent of data, are filtered and replaced with <OOV>
    if feature_reduce and feature_reduce != 0:
        X, lookup_dict = reduce_features(X, feature_reduce, lookup_dict)
        # rearrange lookup dict so that the keys include all values between 0 to len(features)
        lookup_dict = {i: feat for i, feat in enumerate(lookup_dict.values())}

    return X, lookup_dict

def validate_arguments(dataframe, text_col, feature_cols, methods):

    # text_col
    if text_col is None:
        print("No text features included in model")
    elif text_col not in dataframe.columns:
        print("Not valid text column")
        exit(1)

    # feature_col
    if type(feature_cols) != list:
        print("feature_col should be a list of feature column names")
        exit(1)
    if len(feature_cols) > 0:
        if not set(feature_cols).issubset(dataframe.columns):
            print(feature_cols)
            print("To load LIWC/MFD features, load dataframe with \'feature_cols\' as columns")
            exit(1)

    # methods
    if type(methods) != list:
        methods = [methods]

    gen_list = ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]

    for method in methods:
        if method not in gen_list:
            print("{} is not an existing method".format(method))
            exit(1)

def reduce_features(X, feature_reduce, lookup):
    drop_columns = []
    for col in range(X.shape[1]):
        feat, freq = np.unique(X[:,col], return_counts=True)
        if float(max(freq)) / float(X.shape[0]) > 1 - feature_reduce:
            drop_columns.append(col)

    for col in drop_columns:
        print("Dropping columns " + lookup[col])
        lookup.pop(col)
    if type(X) == np.ndarray:
        X = np.delete(X, drop_columns, 1)
    #X1 = sparse.csr_matrix(np.array(X))
    else:
        X = dropcols_coo(X, drop_columns)
    return X, lookup

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()




def tfidf(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir,ngrams,sent_tokenizer, comp_measure = "cosine-sim"):
    return TfidfVectorizer(min_df=10, stop_words='english',
            tokenizer=sent_tokenizer, ngram_range=ngrams), {'alias': 'tfidf'}

def bagofmeans(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    if training_corpus is None or bom_method is None:
        print("Specify bom_method and training_corpus")
    return (BoMVectorizer(training_corpus,
                         embedding_type=bom_method,
                         tokenizer=sent_tokenizer, data_path=data_dir)
                         , {'alias': "_".join([bom_method, training_corpus])})

def ddr(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    if dictionary is None or training_corpus is None or bom_method is None:
        print("Specify dictionary, bom_method, and training_corpus")
        exit(1)
    sim = cosine_similarity if comp_measure == 'cosine-sim' else None
    return (DDRVectorizer(training_corpus,
                         embedding_type=bom_method,
                         tokenizer=sent_tokenizer,
                         data_path=data_dir,
                         dictionary=dictionary,
                         similarity=sim), {'alias': "_".join([bom_method, training_corpus, dictionary])})

def lda(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    num_topics = 100
    return (LDAVectorizer(seed=random_seed,
                         tokenizer=sent_tokenizer,
                         num_topics=num_topics),
           {'alias': "LDA_" + str(num_topics) + "topics"})


def dictionary(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    return (DictionaryVectorizer(data_path= data_dir, dictionary_name= dictionary), {"alias": "Dictionary_" + dictionary})
"""
def ngram(dataframe, text_col, bom_method, training_corpus, dictionary, random_seed, data_dir, ngrams):
    return (NgramVectorizer(n = ngrams, tokenizer= tokenize), {"alias": "Ngram"})
"""

def infersent(dataframe, text_col, bom_method, training_corpus, dictionary,
                random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    return (InfersentVectorizer(data_dir, tokenizer=tokenize), {'alias': "InferSent4096"})

def fasttext(dataframe, text_col, bom_method, training_corpus, dictionary,
                random_seed, data_dir, ngrams, sent_tokenizer, comp_measure = "cosine-sim"):
    return (FastTextVectorizer(data_dir), {'alias': "FastText_wiki"})

