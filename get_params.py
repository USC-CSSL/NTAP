import json
import sys
import os

import pandas as pd

def feature_gen_params():
    params = dict()

    params['data'] = 'indiv.pkl'
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
    if len(sys.argv) != 2:
        print("Usage: python gen_params.py output_name.json")
        exit(1)

    params = feature_gen_params()
    outname = sys.argv[1] if sys.argv[1].endswith('.json') else sys.argv[1] + ".json"
    with open(os.path.join("params", "features", outname), 'w') as fo:
        json.dump(params, fo, indent=4)

