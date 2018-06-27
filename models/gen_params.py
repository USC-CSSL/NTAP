import json

if __name__ == '__main__':
    params = dict()
    params['data_dir'] = '/home/aida/neural_profiles_datadir/'
    params['scoring_dir'] = '/home/aida/PycharmProjects/Neural-Language-Profiles/scoring/'

    # choices: indiv, concat
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'full_dataframe.pkl'

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

    # should be from ["lemmatize", "all_alpha", "link", "hashtag", "stop_words", "emojis", "partofspeech", "stem", "mentions", "ascii"]
    params['preprocessing'] = ["link", "emojis", "mentions", "lemmatize", "stop_words"]
    params['feature_reduce'] = 0

    # [min, max]. default = [0, 1]
    params['ngrams'] = [0, 2]

    with open("params/test_lda.json", 'w') as fo:
        json.dump(params, fo, indent=4)

