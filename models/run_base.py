import os, json
import pandas as pd
import numpy as np

from baselines import Classifier  #, Regressor

param_path = os.environ['PARAMS']
source_path = os.environ['SOURCE_PATH']
feature_path = os.environ['FEAT_PATH']
prediction_path = os.environ['PRED_PATH']

if __name__ == '__main__':
    with open(param_path, 'r') as fo:
        params = json.load(fo)
    source_df = pd.read_pickle(source_path)
    feature_df = pd.read_pickle(feature_path)

    for target in params["target_cols"]:
        print("Predicting {}".format(target))
        missing_indices = list(source_df[source_df[target] == -1.].index)
        target_df = source_df.drop(missing_indices)
        features = feature_df.drop(missing_indices)
        
        X = features.values
        y = target_df[target].values
        feature_names = features.columns.tolist()
        instance_names = list(features.index)

        if params["prediction_task"] == 'classification':
            predictor = Classifier(params)
        elif params["prediction_task"] == 'regression':
            predictor = Regressor(params)
        else:
            raise ValueError("Invalid prediction_task; candidates: classification|regression")
        
        predictions, param_sets, index_dict = predictor.cv_results(X, y)
        row_indices = {k:[instance_names[keys] for keys in v] for k,v in index_dict.items()}
        pred_series = predictor.format_results(predictions, 
                                               y.astype(int), 
                                               row_indices)
        pred_series.to_pickle(prediction_path + target + '.pkl')