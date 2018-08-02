import json

if __name__ == '__main__':
    params = dict()
    root = '/Users/BrendansMac/Documents/'
    params['data_dir'] = root + '/gab_project/data/'
    params['scoring_dir'] = root + '/gab_project/scoring/'

    # choices: indiv, concat
    params['config_text'] = 'indiv'
    #params['dataframe_name'] = 'binned_gab_dataframe.pkl'
    #params['dataframe_name'] = 'gab_network.pkl'
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
    params['dictionary'] = 'moral_foundations_theory'
    params['models'] = ['log_regression']  # ['elasticnet']
    """
    params['targets'] = ['care', 'harm', 'fairness', 'cheating', 'authority',
                         'subversion', 'loyalty', 'betrayal', 'purity',
                         'degradation', 'cv', 'hd', 'moral', 'hate']
    """
    params['targets'] = ['hate']
    #params['targets'] = ['MFQ_' + s + '_AVG' for s in ["FAIRNESS", "INGROUP", "PURITY", "AUTHORITY", "HARM"]]
    params['metrics'] = ['accuracy']
    params['random_seed'] = 51

    # should be from ["lemmatize", "all_alpha", "link", "hashtag", "stop_words", "emojis", "partofspeech", "stem", "mentions", "ascii"]
    params['preprocessing'] = ["link", "mentions"]
    params['feature_reduce'] = 0

    # [min, max]. default = [0, 1]
    params['ngrams'] = [0, 2]
    params['output_name'] = 'lda'

    with open("params/lda.json", 'w') as fo:
        json.dump(params, fo, indent=4)

