import json
import sys
import os

import pandas as pd

def get_params():
    params = dict()
    params['data_dir'] = '/home/aida/neural_profiles_datadir/'
    params['scoring_dir'] = '/home/aida/PycharmProjects/Neural-Language-Profiles/scoring/'

    # choices: indiv, concat
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'yourmorals_df.pkl'

    # choices from ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]
    params['feature_methods'] = ['lda']

    # should be from dataframe's columns
    params['feature_cols'] = []

    # should be one of the dataframe's columns that contains the text
    params['text_col'] = 'fb_status_msg'

    # should be from dataframe's columns
    params['ordinal_cols'] = list()  # ['age']

    # should be from dataframe's columns
    params['categorical_cols'] = list()  # ['gender']

    #
    params['training_corpus'] = 'wiki_gigaword'
    params['embedding_method'] = 'GloVe'
    params['dictionary'] = 'moral_foundations_theory'
    params['models'] = ['elasticnet']
    params['targets'] = ['MFQ_' + s + '_AVG' for s in ["FAIRNESS", "INGROUP", "PURITY", "AUTHORITY", "HARM"]]
    params['metrics'] = ['r2']
    params['random_seed'] = 51

<<<<<<< HEAD
    # should be from ["lemmatize", "all_alpha", "link", "hashtag", "stop_words", "emojis", "partofspeech", "stem", "mentions", "ascii"]
    params['preprocessing'] = ["link", "emojis", "mentions", "lemmatize", "stop_words"]
    params['feature_reduce'] = 0

    # [min, max]. default = [0, 1]
    params['ngrams'] = [0, 2]

    with open("params/test_lda.json", 'w') as fo:
=======
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
>>>>>>> 1b5ca7738a0e698e4cf6cf001e5daad4aed6ea02
        json.dump(params, fo, indent=4)

