import json
import sys
import os

import pandas as pd

def get_baseline_params():
    params = dict()
    params['method'] = 'log_regression'


def mfq_data_params(group_by='post', project='MFQ-facebook'):
    params = dict()
    params['project'] = project
    params['group_by'] = group_by  # 'user'
    return params

def get_data_params(project="MFQ-facebook"):
    if project == "MFQ-facebook":
        group_by = 'post'
        return mfq_data_params(group_by=group_by, project=project)
    else:
        return False

def feature_gen_params():
    params = dict()

    # choices from ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]
    params['feature_methods'] = ['lda']

    # should be one of the dataframe's columns that contains the text
    params['text_col'] ='fb_status_msg'

    params['training_corpus'] = 'wiki_gigaword'
    params['embedding_method'] = 'GloVe'
    params['dictionary'] = 'liwc'
    params['random_seed'] = 51

    # should be from ["lemmatize", "all_alpha", "link", "hashtag", "emojis", "partofspeech", "stem", "mentions", "ascii"]
    params['extract'] = ["link", "mentions", "hashtag", "emojis", "mentions"]
    params['preprocess'] = ['lemmatize', 'stem']
    params['stopwords'] = 'default'  # None, 'my_list.txt', etc.
    params['tokenize'] = 'default'
    # [min, max]: default = [0, 1]
    params['ngrams'] = [0, 2]

    ### Post-processing ###
    ### Develop this part. Type of reduction, etc. ###
    params['feature_reduce'] = 0

    return params

if __name__ == '__main__':
    name = "default"
    params = feature_gen_params()
    outname = os.path.join("params", "features", name + '.json') 
    with open(outname, 'w') as fo:
        json.dump(params, fo, indent=4)
        print("Wrote features params to %s" % outname)
    data_params = get_data_params()
    outname = os.path.join("params", "data", name + '.json')
    with open(outname, 'w') as fo:
        json.dump(data_params, fo, indent=4)
        print("Wrote data params to %s" % outname)
