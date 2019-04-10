
import re, sys, json, os, argparse
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn_pandas import DataFrameMapper
from scipy import sparse, spatial

from baselines.vectorizers.LDA import LDAVectorizer
from baselines.vectorizers.DDR import DDRVectorizer
from baselines.vectorizers.bag_of_means import BoMVectorizer
from baselines.vectorizers.dictionary import DictionaryVectorizer
from baselines.vectorizers.dictionarycount import DictionaryCountVectorizer
from baselines.vectorizers.similarcount import SimilarCountVectorizer
#from baselines.vectorizers.fasttext import FastTextVectorizer
from tokenization.tokenizers import wordpunc_tokenize, happiertokenize, tweettokenize
from nltk.corpus import stopwords

def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

toks = {'happier': happiertokenize,
        'wordpunc': wordpunc_tokenize,
        'tweet': tweettokenize}

stoplists = {'nltk': stopwords.words('english'),
             'sklearn': 'english',
             'None': None}

class Features:
    def __init__(self, dest_dir, params):
        self.dest = os.path.join(dest_dir, "features")
        if not os.path.isdir(self.dest):
            os.makedirs(self.dest)
        self.data = pd.DataFrame()
        self.params = params['feature_params']
        self.path_params = params['path']
        """
        if type(params) == dict:
            self.params = params
        else:
            with open(params, 'r') as fo:
                self.params = json.load(fo)
        """

    def load(self, file_str):
        ending = file_str.split('.')[-1]
        if ending == 'pkl':
            source = pd.read_pickle(file_str)
        elif ending == 'csv':
            source = pd.read_csv(file_str)
        elif ending == 'tsv':
            source = pd.read_csv(file_str, delimiter='\t')
        else:
            raise ValueError("Unsupported file type: {}".format(ending))
        cols = source.columns.tolist()
        #if text_col is None:  #TODO: load text_col from calling file
        text_col = self.__get_text_col(cols)
        self.data[text_col] = source[text_col]
        self.text_col = text_col

    def __get_text_col(self, cols):
        print("...".join(cols))
        notvalid = True
        while notvalid:
            text_col = input("Enter text col from those above: ")
            if text_col.strip() not in cols:
                print("Not a valid column name")
            else:
                notvalid = False
        return text_col
    """
    function: fit
    doc: takes string indicating feature transform, returns list of transformers
         to be used by sklearn-pandas (DataFrameMapper)
    """
    def fit(self, feature):
        self.feature_name = feature
        stopwords = stoplists[self.params["stopwords"]]
        tokenizer = toks[self.params["sent_tokenizer"]]
        if feature == 'tfidf':
            self.transformer = (self.text_col, TfidfVectorizer(
                        max_features=self.params["num_words"], 
                        stop_words=stopwords,
                        tokenizer=tokenizer,
                        ngram_range=self.params["ngrams"]), {'alias': 'tfidf'})

        elif feature == 'bagofmeans':
            self.transformer = (self.text_col,
                            BoMVectorizer(
                                embedding_type=self.params['bom_method'],
                                tokenizer=tokenizer,
                                stop_words=stopwords, glove_path=self.path_params["glove_path"],
                                word2vec_path=self.path_params["word2vec_path"]
                               )
                               , {'alias': self.params["bom_method"]})

        elif feature == 'ddr':
            sim = cosine_similarity 
            self.transformer = (self.text_col,
                                DDRVectorizer(embedding_type=self.params['bom_method'],
                                    tokenizer=tokenizer,
                                    stop_words=stopwords,
                                    dictionary=self.params['dictionary'],
                                    similarity=sim,
                                    dict_path=self.path_params["dictionary_path"],
                                    glove_path=self.path_params["glove_path"],
                                    word2vec_path=self.path_params["word2vec_path"]),
                               {'alias': "_".join([self.params['bom_method'], 
                                                   self.params['dictionary']])})

        elif feature == 'lda':
            self.transformer = (self.text_col,
                                LDAVectorizer(
                                    save_dir=self.dest,
                                    tokenizer=tokenizer,
                                    mallet_path=self.path_params["mallet_path"],
                                    num_topics=self.params["num_topics"],
                                    num_iter=self.params["num_iter"],
                                    seed=self.params["random_seed"],
                                    num_words=self.params["num_words"],
                                    stop_words=stopwords),
                                {'alias': "LDA_" + str(self.params["num_topics"]) + "topics"})
        elif feature == 'dictionary':
            self.transformer = (self.text_col, DictionaryVectorizer(
                        dictionary_name=self.params["dictionary"], dict_path=self.path_params["dictionary_path"]),
                        {"alias": "Dictionary_" + self.params["dictionary"]})
        elif feature == 'dictionarycount':
            self.transformer = (self.text_col, DictionaryCountVectorizer(
                        dictionary_name=self.params["dictionary"], dict_path=self.path_params["dictionary_path"]),
                        {"alias": "Dictionary_" + self.params["dictionary"]})
        elif feature == 'simcount':
            self.transformer = (self.text_col, SimilarCountVectorizer(
                                dictionary_name=self.params["dictionary"],
                                embedding_type=self.params['bom_method'],
                                tokenizer=tokenizer, dict_path=self.path_params["dictionary_path"],
                                glove_path=self.path_params["glove_path"], word2vec_path=self.path_params["word2vec_path"]),
                        {"alias": "Dictionary_" + self.params["dictionary"]})

    """
    transform: assumes that `fit' has been called
    writes features dataframe to file
    """
    def transform(self):

        doc_index = list(self.data.index)
        transformers = [self.transformer]
        mapper = DataFrameMapper(transformers, sparse=False, input_df=True)
        X = mapper.fit_transform(self.data)
        feature_df = pd.DataFrame(X, columns=mapper.transformed_names_)
        feature_df.index = doc_index
        self.__write(feature_df)
    
    """
    features is a dataframe of features, with valid index and column names
    outputs dataframe
    """
    def __write(self, features, formatting='.tsv'):
        fname = os.path.join(self.dest, self.feature_name + formatting)
        if formatting == '.pkl':
            features.to_pickle(fname)
        elif formatting == '.csv':
            features.to_csv(fname)
        elif formatting == '.tsv':
            features.to_csv(fname, sep='\t')
        else:
            raise ValueError("Invalid file format: {}".format(formatting))
        
