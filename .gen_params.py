import json
import sys
import os

import pandas as pd

def get_params():
    params = dict()
    root = '/home/aida/neural_profiles_datadir/'
    params['data_dir'] = root + '/data/'
    params['scoring_dir'] = root + '/scoring/'

    # choices: indiv, concat
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'lda_gab_data.pkl'
    # choices from ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]
    params['feature_methods'] = []

    # should be from dataframe's columns
    params['feature_cols'] = ["topic{}".format(idx) for idx in range(100)]

    # should be one of the dataframe's columns that contains the text
    params['text_col'] = None # 'fb_status_msg'

    # should be from dataframe's columns
    params['ordinal_cols'] = list()  # ['age']

    # should be from dataframe's columns
    params['categorical_cols'] = list()  # ['gender']

    params['training_corpus'] = 'wiki_gigaword'
    params['embedding_method'] = 'GloVe'
    params['dictionary'] = 'liwc'
    #params['models'] = ['log_regression', 'svm']  # ['elasticnet']
    params['metrics'] = ['accuracy']
    params['random_seed'] = 51

    # should be from ["lemmatize", "all_alpha", "link", "hashtag", "stop_words", "emojis", "partofspeech", "stem", "mentions", "ascii"]
    params['preprocessing'] = ["link", "mentions"]
    params['feature_reduce'] = 0

    # [min, max]. default = [0, 1]
    params['ngrams'] = [0, 2]
    params['output_name'] = 'lda'

    # added params for neural methods
    params["models"] = ["lstm"]  # other options: nested_lstm, attention_lstm, ...
    params["num_layers"] = 1
    params["hidden_size"] = 1028
    params["vocab_size"] = 10000
    params["embedding_size"] = 300
    params["batch_size"] = 100
    params["learning_rate"] = 1e-4
    params["pretrain"] = "glove-800"
    params["dropout_ratio"] = 0.5

    return params

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python gen_params.py output_name.json")
        exit(1)
    params = get_params()
    with open(os.path.join("params", sys.argv[1]), 'w') as fo:
        json.dump(params, fo, indent=4)

