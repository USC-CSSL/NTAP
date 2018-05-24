import pandas as pd
import numpy as np
import sys 
import json
import re
import string
import operator
import os

from columns import demographics, MFQ_AVG

from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk_tokenizer = tokenize.TreebankWordTokenizer()

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, CategoricalEncoder
from sklearn.impute import SimpleImputer as Imputer
from sklearn_pandas import gen_features, DataFrameMapper, CategoricalImputer

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens

def prep_text(dataframe, method='alpha'):
    if method == 'alpha':

# returns transformer list, one per generated/loaded text feature
def get_text_transformer(dataframe, 
                         text_col,  # can be list (if loading features) or string (if generating)
                         method,  # type of features to load/generate
                         punc=False,  # remove punctuation in preprocessing
                         clean_type='alpha',  # valid characters for a 'word'
                         bom_method=None,  # options: 'skipgram', 'glove'
                         training_corpus=None,  # options: 'google-news', 'wiki', 'common-crawl'
                         dictionary=None  # options: 'liwc', 'mfd'
                         ):
    # Either generates features from text (tfidf, skipgram, etc.) or load from file

    already_features = type(text_col) == list
    gen_list, load_list = ['tfidf', 'bag-of-means', 'ddr', 'fasttext', 'infersent'], ['dictionary_counting']

    if not already_features:
        if method not in gen_list:
            print("If generating features from text, specify \'text_col\' as a string object")
            exit(1)
        if text_col not in dataframe.columns:
            print("Not valid text column")
            exit(1)
    if already_features:
        if method not in load_list:
            print("If loading features from text, specify \'text_col\' as a list (of columns)")
            exit(1)
        if not set(text_col).issubset(dataframe.columns):
            print("To load LIWC/MFD features, load dataframe with \'text_col\' as columns")
            exit(1)

    if method == 'tfidf':
        # process text
        transformers.append((feat, [TfidfVectorizer(min_df=10, 
                                                   stop_words='english',
                                                   preprocessor=process_text,
                                                   tokenizer=tokenize),
                                    StandardScaler]
                            ))
    elif method == 'dictionary_counting':
        if len(text_col) > 0:
            transformer_list += gen_features(
                    columns=[ [col] for col in text_col],
                    classes=[StandardScaler])

    elif method == 'bag-of-means':
        if training_corpus is None or bom_method is None:
            print("Specify bom_method and training_corpus")
            exit(1)
        transformers.append((text_col, BoMVectorizer(training_corpus,
                                                     remove_stopwords=True,
                                                     preprocessor=process_text,
                                                     tokenizer=tokenize)))
    elif method == 'ddr':
        if dictionary is None or training_corpus is None or bom_method is None:
            print("Specify dictionary, bom_method, and training_corpus")
            exit(1)
        transformers.append((text_col, [BoMVectorizer(training_corpus,
                                                      remove_stopwords=True,
                                                      preprocessor=process_text,
                                                      tokenizer=tokenize),
                                        DDRVectorizer(dictionary=dictionary,
                                                      similarity='cosine-sim'),
                                        StandardScaler
                                   ]))
        # chain skipgram-bom with liwc transformer
    elif method == 'fasttext':  # exceeds at syntactic tasks
        # call fasttext executable with subprocess module
    elif method == 'infersent':
        transformers.append((feat, InferSentVectorizer(args)))
    else:
        print("Invalid method || exiting")
        exit(1)
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python baselines.py Path/To/DataDir/")
    exit(1)

scoring_dir = "../scoring/"

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

data_dir = sys.argv[1]
config_text =  'indiv'  # 'concat'

if config_text == 'concat':
    df = pd.read_pickle(data_dir + '/' + "concat_df.pkl")
else:
    df = pd.read_pickle(data_dir + '/' + "full_dataframe.pkl")

df = df.reindex()
print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

# feature_versions format: {name: (text_col, punc, [ord_cols], [cat_cols]}
feature_versions = {'tfidf_rawtext': ('fb_status_msg', 
                                      False,
                                      list(),
                                      list()
                                     ),
                    'tfidf_rawtext_demog': ('fb_status_msg',
                                            False,
                                            ['age', 'gender'],
                                            list()
                                           )
                   }
                
                # }   ,
                #'lemmatized_nopunc': ('lemmatized_posts', False),
                #'raw_text_punc': ('fb_status_msg', True),
                #'lemmatized_punc': ('lemmatized_posts', True)}


models = ["elasticnet"]  # , "GBRT"]
targets = MFQ_AVG
metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
seed = 51
num_words = 10000

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

alpha_re = re.compile(r"[^a-zA-Z\s]")
length_re = re.compile(r'\w{3,}')

for feat in feature_versions:
    text_column, punc_status, ordinal_cols, categorical_cols = feature_versions[feat]
    text = df[text_column].values.tolist() 
    text = [" ".join(length_re.findall(alpha_re.sub('', doc))) for doc in text]
    #if not punc_status:
    #    table = str.maketrans(dict.fromkeys(string.punctuation))  # remove punctuation
    #    text = [doc.translate(table) for doc in text]
    df[feat] = pd.Series(text, index=df.index)
    if len(categorical_cols) > 0: 
        transformer_list += gen_features(
                columns=[ [col] for col in categorical_cols],
                classes=[CategoricalImputer,
                         {'class': CategoricalEncoder
                         }
                        ])
    if len(ordinal_cols) > 0:
        transformer_list += gen_features(
                columns=[ [col] for col in ordinal_cols],
                classes=[{'class': Imputer, 'missing_values':-1},
                         {'class': MinMaxScaler}
                        ])
    
    print("Performing data transform for {} text features, and \n[{}] categorical columns, and \n[{}] ordinal columns".format(
            feat, ",".join(categorical_cols), ",".join(ordinal_cols)))

    mapper = DataFrameMapper(transformer_list, sparse=True)
    X = mapper.fit_transform(df)
    lookup_dict = {i:feat for i, feat in enumerate(mapper.transformed_names_)}
    
    scoring_dict = dict()

    for col in targets:
        print("Working on predicting {}".format(col))
        scoring_dict[col] = dict()
        Y = df[col].values.tolist()
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        for model in models:
            if model == 'bow_elasticnet':
                scoring_dict[col][model] = dict()
                regressor = SGDRegressor(loss='squared_loss',  # default
                                        penalty='elasticnet',
                                        max_iter=50,
                                        shuffle=True,
                                        random_state=seed,
                                        verbose=0
                                        )
                choose_regressor = GridSearchCV(regressor, cv=kfold, iid=True, 
                                                param_grid={"alpha": 10.0**-np.arange(1,7), 
                                                            "l1_ratio": np.arange(0.15,0.25,0.05)
                                                        }
                                               )

                choose_regressor.fit(X,Y)
                best_model = choose_regressor.best_estimator_
                scoring_dict[col][model]['params'] = choose_regressor.best_params_
                coef_dict = {i:val for i,val in enumerate(best_model.coef_)}
                word_coefs = {lookup_dict[i]:val for i, val in coef_dict.items()}
                abs_val_coefs = {word:abs(val) for word, val in word_coefs.items()}
                top_features = sorted(abs_val_coefs.items(), key=operator.itemgetter(1), reverse=True)[:100]
                real_weights = [[word, word_coefs[word]] for word, _ in top_features]
                scoring_dict[col][model]['top_features'] = real_weights
            for metric in ['r2']:
                results = best_model.score(X, Y)
                scoring_dict[col][model][metric + "_mean"] = "{0:.3f}".format(results.mean())
                scoring_dict[col][model][metric + "_std"] = "{0:.3f}".format(results.std())

    scoring_output = os.path.join(scoring_dir, config_text, feat)
    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, "scores_full" + ".json")
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
