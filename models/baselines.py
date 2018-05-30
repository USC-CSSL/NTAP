import pandas as pd
import sys, os, json

from columns import demographics, MFQ_AVG
from make_features import get_text_transformer
from preprocess import preprocess_text
from evaluate import evaluate_models

from sklearn_pandas import DataFrameMapper


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
        preprocess_methods = params["preprocessing"]

        feature_methods = params["feature_method"]  # a list of methods. options: bag-of-means, ddr, fasttext, infersent
        text_col = params["text_col"]  # the column that contains the raw text
        feature_col = params["feature_col"] # name of the columns that are considered as features. These features are already extracted and exist in the dataframe
        ordinal_cols = params["ordinal_features"]
        categorical_cols = params["categorical_features"]

        # for doing DDR or BoM feature generation
        corpus = params['training_corpus'] # options: wiki, common_crawl, twitter, custom
        word2vec_method = params['embedding_method']  # options: skipgram, glove, cbow
        dictionary = params['dictionary']
    except KeyError:
        print("Could not load all parameters; if you're not using a parameter, set it to None")
        exit(1)

    # Reading the dataset
    df = pd.read_pickle(data_dir + '/' + dataframe_fname)
    print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

    #Preprocessing the data
    preprocess_text = preprocess_text(df, text_col, preprocess_methods ,data_dir)

    # Transform features
    transformer_list = get_text_transformer(df, text_col, feature_methods, feature_col, ordinal_cols, categorical_cols,
                                            bom_method=word2vec_method, training_corpus=corpus, dictionary=dictionary)

    print(transformer_list)

    mapper = DataFrameMapper(transformer_list, sparse=True, input_df=True)
    X = mapper.fit_transform(df)
    lookup_dict = {i: feat for i, feat in enumerate(mapper.transformed_names_)}

    # Performing classification
    scoring_dict = evaluate_models(df, X, targets, lookup_dict, models, seed)

    scoring_output = os.path.join(scoring_dir, config_text, "-".join(f for f in feature_methods))
    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, "scores_full" + ".json")
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
