import os, json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from baselines import Classifier, Regressor

param_path = os.environ['PARAMS']
source_dir = os.environ['SOURCE_DIR']
feature_dir = os.environ['FEAT_DIR']
prediction_path = os.environ['PRED_DIR']

if __name__ == '__main__':
    with open(param_path, 'r') as fo:
        params = json.load(fo)
    
    source_df = pd.read_pickle(os.path.join(source_dir, params['group_by'] + '.pkl'))
    feature_df = pd.read_pickle(os.path.join(feature_dir, params['group_by'] + '.pkl'))

    for target in params["target_cols"]:
        print("Predicting {}".format(target))
        missing_indices = list(source_df[source_df[target] == -1.].index)
        target_df = source_df.drop(missing_indices)
        features = feature_df.drop(missing_indices)
       
        X = features.values
        y = target_df[target].values
        plt.hist(y, color='blue', edgecolor='black')
        plt.savefig("thing.png")
        feature_names = features.columns.tolist()
        instance_names = list(features.index)

        if params["prediction_task"] == 'classification':
            predictor = Classifier(params)
        elif params["prediction_task"] == 'regression':
            predictor = Regressor(params)
        else:
            raise ValueError("Invalid prediction_task; candidates: classification|regression")
        
        predictions, param_sets, index_dict, features = predictor.cv_results(X, y)
        row_indices = {k:[instance_names[keys] for keys in v] for k,v in index_dict.items()}
        pred_series = predictor.format_results(predictions, 
                                               y, 
                                               row_indices)
        if not os.path.exists(os.path.join(prediction_path, target)):
            os.makedirs(os.path.join(prediction_path, target))
        pred_series.to_pickle(os.path.join(prediction_path + target, 'predictions.pkl'))

        model_features = predictor.format_features(features,
                                               feature_names)
        model_features.to_pickle(os.path.join(prediction_path + target, 'features.pkl'))
