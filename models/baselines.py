import pandas as pd
import numpy as np
import sys 
import json
import re
import string
import operator
import os

from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk_tokenizer = tokenize.TreebankWordTokenizer()

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens

def get_tfidf(dataframe):
    print("Creating feature vectors")
    table = str.maketrans(dict.fromkeys(string.punctuation))  # remove punctuation
    X_docs = [text.translate(table) for text in dataframe[text_column].values.tolist()]
    #X_docs = dataframe[text_column].values.tolist()
    vect = TfidfVectorizer(tokenizer=tokenize, min_df=10, stop_words='english', lowercase=True)
    X = vect.fit_transform(X_docs)
    term_to_indices = vect.vocabulary_
    indices_to_terms = {value:key for key,value in term_to_indices.items()}
    return X, indices_to_terms

if len(sys.argv) != 2:
    print("Usage: python baselines.py Path/To/DataDir/")
    exit(1)

data_dir = sys.argv[1]
config_text = 'individual' # 'concat'
text_version = 'raw'  # options: ['raw', 'lemmatized']

if config_text == 'concat':
    df = pd.read_pickle(data_dir + '/' + "concat_df.pkl")
else:
    df = pd.read_pickle(data_dir + '/' + "full_dataframe.pkl")

print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

if text_version == 'raw':
    text_column = "fb_status_msg"
else:
    text_column = "lemmatized_posts"

# models = ["bow_elasticnet", "bow_GBRT", "bom_elasticnet", "bom_GBRT", "infersent_elasticnet", "infersent_GBRT"]

models = ["bow_elasticnet"]

mfq_cols = [col for col in df.columns.tolist() if col.startswith("MFQ") and col.endswith("AVG")]

seed = 7
metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
n = df.shape[0]
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
X, lookup_dict = get_tfidf(df)

scoring_dict = dict()

for col in mfq_cols:
    print("Working on {}".format(col))
    scoring_dict[col] = dict()
    Y = df[col].values.tolist()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    for model in models:
        if model == 'bow_elasticnet':
            scoring_dict[col][model] = dict()
            regressor = SGDRegressor(loss='squared_loss',
                                    penalty='elasticnet',
                                    alpha=1e-4,
                                    l1_ratio=0.15,
                                    max_iter=np.ceil(10**6 ),
                                    shuffle=True,
                                    random_state=seed,
                                    )
            choose_regressor = GridSearchCV(regressor, cv=kfold, 
                                            param_grid={"alpha": 10.0**-np.arange(1,7), 
                                                        "l1_ratio": np.arange(0.05,1.0,20)
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

scoring_output = os.path.join(data_dir, text_version, config_text)
if not os.path.exists(scoring_output):
    os.makedirs(scoring_output)
scoring_output = os.path.join(scoring_output, "scores.json")
with open(scoring_output, 'w') as fo:
    json.dump(scoring_dict, fo, indent=4)
