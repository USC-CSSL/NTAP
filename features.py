
import re, sys, json, os, argparse
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn_pandas import gen_features, CategoricalImputer, DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse

from vectorizers.LDA import LDAVectorizer
from vectorizers.DDR import DDRVectorizer
from vectorizers.bag_of_means import BoMVectorizer
from vectorizers.dictionary import DictionaryVectorizer
from vectorizers.fasttext import FastTextVectorizer

from cossim import cosine_similarity
from tokenization.tokenizers import wordpunc_tokenize, happiertokenize, tweettokenize

toks = {'happier': happiertokenize,
        'wordpunc': wordpunc_tokenize,
        'tweet': tweettokenize}

from nltk.corpus import stopwords

stoplists = {'nltk': stopwords.words('english'),
             'default': 'english'}

from parameters import features as params

def tfidf(**kwargs):
    return TfidfVectorizer(max_features=kwargs["num_words"], 
                           stop_words=kwargs["stop_words"],
                           tokenizer=kwargs["sent_tokenizer"], 
                           ngram_range=kwargs["ngrams"]), {'alias': 'tfidf'}

def bagofmeans(**kwargs):
    return (BoMVectorizer(embedding_type=kwargs['bom_method'],
                          tokenizer=kwargs['sent_tokenizer'], 
                          stop_words=kwargs['stop_words']
                          )
            , {'alias': kwargs["bom_method"]})

def ddr(**kwargs): 
    sim = cosine_similarity if kwargs['comp_measure'] == 'cosine-sim' else None
    return (DDRVectorizer(embedding_type=kwargs['bom_method'],
                          tokenizer=kwargs['sent_tokenizer'],
                          stop_words=kwargs['stop_words'],
                          dictionary=kwargs['dictionary'],
                          similarity=sim), 
            {'alias': "_".join([kwargs['bom_method'], kwargs['dictionary']])})

def lda(**kwargs):
    return (LDAVectorizer(seed=kwargs["random_seed"],
                          tokenizer=kwargs["sent_tokenizer"],
                          num_topics=kwargs["num_topics"],
                          num_iter=kwargs["num_iter"],
                          num_words=kwargs["num_words"],
                          stop_words=kwargs["stop_words"],
                          ngrams=kwargs["ngrams"]),
            {'alias': "LDA_" + str(num_topics) + "topics"})

def dictionary(**kwargs):
    return (DictionaryVectorizer(dictionary_name=kwargs["dictionary"]), 
            {"alias": "Dictionary_" + dictionary})

def fasttext(**kwargs):
    return (FastTextVectorizer(tokenizer=kwargs["sent_tokenizer"],
                               stop_words=kwargs["stop_words"]), 
            {'alias': "FastText_wiki"})

def load_params(f):
    with open(f, 'r') as fo:
        return json.load(fo)

def collect_features(dataframe):

    text_col, target_col = "text", "target"

    doc_index = list(dataframe.index)
    sent_tokenizer = toks[params["tokenizer"]]
    try:
        stopword_list = stoplists[params["stopwords"]]
    except KeyError:
        stopword_list = None

    transformers = list()
    for method in params['features']:
        transformation = globals()[method](bom_method=params['wordvec'], 
                                           dictionary=params['dictionary'], 
                                           random_seed=os.environ['random_seed'], 
                                           ngrams=params['ngrams'], 
                                           sent_tokenizer=sent_tokenizer,
                                           comp_measure="cosine-sim",
                                           num_topics=params["num_topics"],
                                           num_iter=params["num_iter"],
                                           num_words=params["vocab_size"],
                                           stop_words=stopword_list
                                           )
        transformers.append((text_col, ) + transformation) if type(transformation) == tuple else (text_col, transformation)

    """
    if len(params['feature_cols']) > 0:  # continuous variables
        transformers += gen_features(
                columns=[ [col] for col in params['feature_cols']])
                #classes=[StandardScaler])
    if len(params['categoricals']) > 0:  # categorical variables
        transformers += gen_features(
                    columns=[ col for col in params['categoricals']],
                    classes=[CategoricalImputer, LabelBinarizer]
                            )
    """

    mapper = DataFrameMapper(transformers, sparse=False, input_df=True)
    X = mapper.fit_transform(dataframe)
    lookup_dict = {i: feat for i, feat in enumerate(mapper.transformed_names_)}

    feature_df = pd.DataFrame(X, columns=mapper.transformed_names_)
    feature_df.index = doc_index
    return feature_df

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Processed input data filepath")
    parser.add_argument("--output", help="Destination filepath for features")
    args = parser.parse_args()
    
    data = pd.read_pickle(args.input)
    feature_df = collect_features(data)
    feature_df.to_pickle(args.output)

