import pandas as pd
import numpy as np
import sys 
import json
import re
import operator
import os

from make_features import get_text_transformer, add_categorical

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn_pandas import DataFrameMapper

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python baselines.py params.json")
        exit(1)

    # load parameters from (generated) json file. See 'gen_params.py' 
    with open(sys.argv[1], 'r') as fo:
        params = json.load(fo)

    try:
        data_dir = params["data_dir"]
        scoring_dir = params["scoring_dir"]  # ../scoring_dir/
        config_text = params["config_text"] # indiv
        dataframe_fname = params["dataframe_name"]
        # if loading features, make sure there are denoted as columns in the dataframe
        models = params["models"]  # ['elasticnet', 'GBRT']
        targets = params["targets"]
        metrics = params["metrics"]
        seed = params["random_seed"]

        feature_method = params["feature_method"]  # options: bag-of-means, load_features, ddr, fasttext, infersent
        text_col = params["text_col"]  # if feature_method == 'load_features' then type(text_col) is list
        ordinal_cols = params["ordinal_features"]  
        categorical_cols = params["categorical_features"]

        # for doing DDR or BoM feature generation
        corpus = params['training_corpus'] # options: wiki, common_crawl, twitter, custom
        word2vec_method = params['embedding_method']  # options: skipgram, glove, cbow
        dictionary = params['dictionary']
    except KeyError:
        print("Could not load all parameters; if you're not using a parameter, set it to None")
        exit(1)

    #-----------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------

    df = pd.read_pickle(data_dir + '/' + dataframe_fname)
    print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

    #-----------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------

    # Transform features
    transformer_list = get_text_transformer(df, data_dir, text_col, feature_method, 
                                            bom_method=word2vec_method,
                                            training_corpus=corpus, dictionary=dictionary,
                                            comp_measure='cosine-sim')
    transformer_list = add_categorical(transformer_list, ordinal_cols, categorical_cols)

    mapper = DataFrameMapper(transformer_list, sparse=True, input_df=True)
    X = mapper.fit_transform(df)
    lookup_dict = {i:feat for i, feat in enumerate(mapper.transformed_names_)}
    scoring_dict = dict()

    for col in targets:
        print("Working on predicting {}".format(col))
        scoring_dict[col] = dict()
        Y = df[col].values.tolist()
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        for model in models:
            if model == 'elasticnet':
                scoring_dict[col][model] = dict()
                regressor = SGDRegressor(loss='squared_loss',  # default
                                        penalty='elasticnet',
                                        max_iter=1,
                                        shuffle=True,
                                        random_state=seed,
                                        verbose=1
                                        )
                choose_regressor = GridSearchCV(regressor, cv=kfold, iid=True, 
                                                param_grid={"alpha": 10.0**-np.arange(1,7), 
                                                            "l1_ratio": np.arange(0.15,0.25,0.05)
                                                        }
                                               )

                choose_regressor.fit(X,Y)
                best_model = choose_regressor.best_estimator_
                scoring_dict[col][model]['params'] = choose_regressor.best_params_
                coef_dict = {i:val for i,val in enumerate(best_model.coef_)}
                word_coefs = {lookup_dict[i]:val for i, val in coef_dict.items()}
                abs_val_coefs = {word:abs(val) for word, val in word_coefs.items()}
                top_features = sorted(abs_val_coefs.items(), key=operator.itemgetter(1), reverse=True)[:100]
                real_weights = [[word, word_coefs[word]] for word, _ in top_features]
                scoring_dict[col][model]['top_features'] = real_weights
            
            if model == 'GBRT':
                print("GBRT Model")
                continue
                # Do GBRT shit
                # Post conditions: best_model is the best model (by CV); scoring_dict[col][model] is updated

            
            for metric in ['r2']:
                results = best_model.score(X, Y)
                scoring_dict[col][model][metric + "_mean"] = "{0:.3f}".format(results.mean())
                scoring_dict[col][model][metric + "_std"] = "{0:.3f}".format(results.std())
            
    scoring_output = os.path.join(scoring_dir, config_text, feature_method)
    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, sys.argv[1]) 
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
