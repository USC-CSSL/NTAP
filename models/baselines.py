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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from nltk.stem.porter import PorterStemmer

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens


class TextSelector(BaseEstimator, TransformerMixin):
    """Select the column of text (raw or lemmatized) and transform/process it for Tfidf"""

    def __init__(self, text_column, punc=False):
        self.text_column = text_column
        self.punc = punc

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        if not self.punc:
            table = str.maketrans(dict.fromkeys(string.punctuation))  # remove punctuation
            docs = [text.translate(table) for text in dataframe[self.text_column].values.tolist()]
        else:
            docs = dataframe[self.text_column].values.tolist()
        return docs 


class SubSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        return dataframe.loc[:, self.keys].to_dict('records')

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python baselines.py Path/To/DataDir/")
    exit(1)

scoring_dir = "../scoring/"

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

data_dir = sys.argv[1]
config_text = 'individual' # 'concat'

if config_text == 'concat':
    df = pd.read_pickle(data_dir + '/' + "concat_df.pkl")
else:
    df = pd.read_pickle(data_dir + '/' + "full_dataframe.pkl")

print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


text_versions = {'raw_text_nopunc': ('fb_status_msg', False),
                'lemmatized_nopunc': ('lemmatized_posts', False),
                'raw_text_punc': ('fb_status_msg', True),
                'lemmatized_punc': ('lemmatized_posts', True)}
# models = ["bow_elasticnet", "bow_GBRT", "bom_elasticnet", "bom_GBRT", "infersent_elasticnet", "infersent_GBRT"]
models = ["bow_elasticnet", "bow_GBRT"]
mfq_cols = MFQ_AVG
metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
seed = 7


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


for feat in text_versions:
    text_column, punc_status = text_versions[feat]

    features = FeatureUnion(
            transformer_list=[
                ('text', Pipeline([
                    ('selector', TextSelector(text_column, punc=punc_status)),
                    ('tfidf', TfidfVectorizer(min_df=10, 
                                              stop_words='english', 
                                              lowercase=True, 
                                              tokenizer=tokenize)
                                              ),
                ])),
                ('demographics', Pipeline([
                    ('selector', SubSelector(keys=demographics)),
                    ('demo_feat', DictVectorizer()),
                    ('scaled_demo', StandardScaler(with_mean=False)),
                ])),
            ],
           )
    X = features.fit_transform(df)
    lookup_dict = {i:feat for i, feat in enumerate(features.get_feature_names())}
    
    scoring_dict = dict()

    for col in mfq_cols:
        print("Working on {}".format(col))
        scoring_dict[col] = dict()
        Y = df[col].values.tolist()
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        for model in models:
            if model == 'bow_elasticnet':
                scoring_dict[col][model] = dict()
                regressor = SGDRegressor(loss='squared_loss',  # default
                                        penalty='elasticnet',
                                        tol=1e-4,
                                        shuffle=True,
                                        random_state=seed,
                                        verbose=1
                                        )
                choose_regressor = GridSearchCV(regressor, cv=kfold, 
                                                param_grid={"alpha": 10.0**-np.arange(1,7), 
                                                            "l1_ratio": np.arange(0.0,1.0,0.2)
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
            for metric in metrics:
                results = model_selection.cross_val_score(best_model, X, Y, cv=kfold, scoring=metric)
                scoring_dict[col][model][metric + "_mean"] = "{0:.3f}".format(results.mean())
                scoring_dict[col][model][metric + "_std"] = "{0:.3f}".format(results.std())

    scoring_output = os.path.join(scoring_dir, text_version, config_text)
    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, "scores.json")
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
