import json

if __name__ == '__main__':
    params = dict()
    params['data_dir'] = '/home/brendan/Data/MediaDiet/data/'
    params['scoring_dir'] = '/home/brendan/Data/MediaDiet/scoring/'

    # choices: indiv, concat
    #params['config_text'] = 'indiv'
    params['dataframe_name'] = 'tv_series_raw.pkl'
    # choices from ['tfidf', 'lda', 'bagofmeans', 'ddr', 'fasttext', 'infersent', "dictionary"]
    params['feature_methods'] = ['dictionary']

    # should be from dataframe's columns
    params['feature_cols'] = [] # ["topic{}".format(idx) for idx in range(100)]

    # should be one of the dataframe's columns that contains the text
    params['text_col'] = 'text' # 'fb_status_msg'

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

    with open("params/get_liwc_features.json", 'w') as fo:
        json.dump(params, fo, indent=4)

