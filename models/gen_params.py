import json
import sys
import os

import pandas as pd

def get_params():
    params = dict()
    params['data_dir'] = '/home/brendan/neural_profiles_datadir/'
    params['scoring_dir'] = '/home/brendan/NeuralLanguageProfiles/scoring/'
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'yourmorals_df.pkl'

    params['feature_methods'] = ['fasttext']
    params['feature_cols'] = []
    params['text_col'] = 'fb_status_msg'
    params['ordinal_cols'] = list()  # ['age']
    params['categorical_cols'] = list()  # ['gender']
    params['training_corpus'] = 'wiki_gigaword'
    params['embedding_method'] = 'GloVe'
    params['dictionary'] = 'moral_foundations_theory'
    params['models'] = ['elasticnet']
    params['targets'] = ['MFQ_' + s + '_AVG' for s in ["FAIRNESS", "INGROUP", "PURITY", "AUTHORITY", "HARM"]]
    params['metrics'] = ['r2']
    params['random_seed'] = 51
    params['preprocessing'] = []
    params['feature_reduce'] = 0.001
    params['ngrams'] = [1]

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

