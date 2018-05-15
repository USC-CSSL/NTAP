import pandas as pd
import sys 
import json
import re

from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk_tokenizer = tokenize.TreebankWordTokenizer()

from sklearn.linear_model import ElasticNet

if len(sys.argv) != 2:
    print("Usage: python predict_mfq Path/To/DataDir/")
    exit(1)

data_dir = sys.argv[1]
dataframe = pd.read_pickle(data_dir + '/' + "full_dataframe.pkl")
with open(data_dir + '/' + "users.json", 'r', encoding='utf-8') as fo:
    users = [user_id for user_id, texts in json.load(fo).items()]

models = ["bow_elasticnet", "bow_GBRT", "bom_elasticnet", "bom_GBRT", "infersent_elasticnet", "infersent_GBRT"]
config_text = 'concat'


"""
To-Do
    - Specify config (concat all text vs. individual document)
    - Options: 
        - Discard posts with less than 'k' tokens
    - Tokenize text (via nltk)
    - Vectorize (BoW) text (TfidfVectorizer via sklearn)   
    - Build models
        - Regression (with elasticNet)
        - GBRT
"""

print("Full dataframe has {} rows and {} columns".format(dataframe.shape[0], dataframe.shape[1]))

mfq_cols = [col for col in dataframe.columns.tolist() if col.startswith("MFQ") and col.endswith("AVG")]
for col in mfq_cols:
    print("Working on {}".format(col))
    sub_df = dataframe[dataframe[col] != -1]
    X_docs = list()
    Y_labels = list()
    if config_text == 'concat':
        for user in users:
            texts = list()
            for i, row in sub_df.iterrows():
                if str(row["userid"]) == str(user):
                    texts.append(row["fb_status_msg"])
            if len(texts) == 0:
                continue
            doc = "\n".join(texts)
            X_docs.append(doc)
            Y_labels.append(str(user))
        vect = TfidfVectorizer(tokenizer=nltk_tokenizer, stopwords='english', lowercase=True)
        X_train = vect.fit_transform(X_docs)
        en_model = ElasticNet()
        en_model.fit(X_train, Y_labels)
        print(en_model.coef_)
        
