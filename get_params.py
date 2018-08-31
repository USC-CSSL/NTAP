import json
import sys
import os

import pandas as pd

def add_baseline_params(params):
    params['prediction_task'] = 'classification'  # regression
    params['prediction_method'] = 'log_regression'
    params['target_cols'] = ['MFQ_cruel', 'MFQ_compassion']
    params['k_folds'] = 3

def add_data_params(params, project="MFQ-facebook"):
    params['text_col'] = 'fb_status_msg'
    params['extract'] = []  # ["link", "mentions", "hashtag"]  # "emojis"
    params['preprocess'] = [] # ['stem']  # 'lemmatize'
    ### Working: link, mentions, hashtag, stem
    ### Not Working: lemmatize, emojis
    params['lower'] = True
    params['stopword_list'] = 'nltk'  # None, 'my_list.txt', etc.
    params['entity_linking'] = False
    params['rho'] = 0.1  # Suggested rho value between 0.1 and 0.3
    if project == "MFQ-facebook":
        params['group_by'] = 'post'

def add_feature_params(params):
    # choices from ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]
    params['feature_methods'] = ['ddr']

    # should be one of the dataframe's columns that contains the text
    params['text_col'] ='fb_status_msg'
    params['feature_cols'] = list()
    params['categoricals'] = list()

    params['word_embedding'] = 'glove'  # word2vec
    params['dictionary'] = 'mfd'
    params['random_seed'] = 51

    params['tokenize'] = 'wordpunc'
    # [min, max]: default = [0, 1]
    params['ngrams'] = [0, 2]

    ### Post-processing ###
    ### Develop this part. Type of reduction, etc. ###
    params['feature_reduce'] = 0

def add_neural_params(params):
    params["learning_rate"] = 0.0001
    params["batch_size"] = 100
    params["keep_ratio"] = 0.66

    #choose from ["GRU", "LSTM"]
    params["cell"] = "LSTM"

    #choose from ["LSTM", "BiLSTM", "CNN", "RNN"]
    params["model"] = "LSTM"
    params["vocab_size"] = 10000
    params["embedding_size"] = 300
    params["hidden_size"] = 256
    params["pretrain"] = True
    params["train_embedding"] = False
    params["num_layers"] = 1
    params["n_outputs"] = 6

    #choose from ["Mean", "Weighted"]
    params["loss"] = "Mean"
    params["epoch"] = 500
    params["save_vectors"] = False

    # should be dataframe columns that are needed to be visualized with embedding vectors
    params["visualize_cols"] = []

if __name__ == '__main__':
    instance_name = os.environ["INSTANCE_NAME"]
    proj_name = os.environ["PROJ_NAME"]
    params = dict()
    add_data_params(params, project=proj_name)
    add_feature_params(params)
    add_baseline_params(params)
    add_neural_params(params)
    outname = os.path.join("params", instance_name + '.json') 
    with open(outname, 'w') as fo:
        json.dump(params, fo, indent=4)
        print("Wrote baseline params to %s" % outname)
