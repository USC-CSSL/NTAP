import json

if __name__ == '__main__':
    params = dict()
    params['data_dir'] = '/home/brendan/neural_profiles_datadir/'
    params['scoring_dir'] = '/home/brendan/NeuralLanguageProfiles/scoring/'
    params['config_text'] = 'indiv'
    params['dataframe_name'] = 'full_dataframe.pkl'

    params['feature_method'] = 'tfidf'
    params['text_col'] = 'fb_status_msg'
    params['ordinal_features'] = list()  # ['age']
    params['categorical_features'] = list()  # ['gender']
    params['training_corpus'] = 'GoogleNews3B'
    params['embedding_method'] = 'skipgram'
    params['dictionary'] = 'liwc'
    params['models'] = ['elasticnet', 'GBRT']
    params['targets'] = ['MFQ_' + s + '_AVG' for s in ["FAIRNESS", "INGROUP", "PURITY", "AUTHORITY", "HARM"]]
    params['metrics'] = ['r2']
    params['random_seed'] = 51


    with open("standard_params.json", 'w') as fo:
        json.dump(params, fo, indent=4)

