import pandas as pd
import sys 
import json
import re
import string

from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk_tokenizer = tokenize.TreebankWordTokenizer()

from sklearn.linear_model import SGDRegressor
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

if len(sys.argv) != 2:
    print("Usage: python predict_mfq.py Path/To/DataDir/")
    exit(1)

config_text = 'concat'

data_dir = sys.argv[1]
if config_text == 'concat':
    dataframe = pd.read_pickle(data_dir + '/' + "concat_df.pkl")
else:
    dataframe = pd.read_pickle(data_dir + '/' + "full_dataframe.pkl")
models = ["bow_elasticnet", "bow_GBRT", "bom_elasticnet", "bom_GBRT", "infersent_elasticnet", "infersent_GBRT"]

"""
To-Do
    - Build models
        - GBRT
"""

print("Dataframe has {} rows and {} columns".format(dataframe.shape[0], dataframe.shape[1]))

mfq_cols = [col for col in dataframe.columns.tolist() if col.startswith("MFQ") and col.endswith("AVG")]

table = str.maketrans(dict.fromkeys(string.punctuation))  # remove punctuation
X_docs = [text.lower().translate(table) for text in dataframe["fb_status_msg"].values.tolist()[:100]]
vect = TfidfVectorizer(tokenizer=tokenize, max_features=10000, stop_words='english', lowercase=True)
print("Creating feature vectors")
X = vect.fit_transform(X_docs)

seed = 7
for col in mfq_cols:
    print("Working on {}".format(col))
    Y = dataframe[col].values.tolist()[:100]
    en_model = SGDRegressor(penalty='elasticnet', verbose=0, max_iter=1000 )
    print("Cross-validating ElasticNet model")

    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    scoring = 'r2'
    results = model_selection.cross_val_score(en_model, X, Y, cv=kfold, scoring=scoring)
    print(results)
    print("R^2: %.3f (%.3f)") % (results.mean(), results.std())

