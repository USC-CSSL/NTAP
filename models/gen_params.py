import json

if __name__ == '__main__':
    params = dict()
    params['data_dir'] = '/home/brendan/neural_profiles_datadir/'
    params['scoring_dir'] = '/home/brendan/NeuralLanguageProfiles/scoring/'
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'full_dataframe.pkl'

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

    with open("test_fasttext.json", 'w') as fo:
        json.dump(params, fo, indent=4)

