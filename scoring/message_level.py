import json
import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
"""
To Do:
    - For each CV: 
        - Compute accuracy (k_hat == k)
        - Compute MSE
    - Rank each message by average accuracy and MSE (get hard-to-classify and easy-to-classify)
"""

def calc_accuracy(predictions):

    cv_metrics = list()
    for cv_fold, cv_df in predictions.groupby(level=0):
        # accuracy
        num_correct, base_correct = 0, 0
        scores = list()
        for mess_id, row in cv_df.iterrows():
            if row['y'] == row['y_hat']:
                num_correct += 1
            if row['y'] == row['y_baseline']:
                base_correct += 1
        perf = ( 1. * num_correct) / len(cv_df)
        base = ( 1. * base_correct) / len(cv_df)
        cv_metrics.append({"acc": perf, "base": base})
    return cv_metrics


def calc_r2(predictions):

    cv_metrics = list()
    for cv_fold, cv_df in predictions.groupby(level=0):
        # r2
        y = cv_df['y'].values
        y_hat = cv_df['y_hat'].values

        r = pearsonr(y, y_hat)

        sse = np.sum( (y - y_hat) ** 2)
        sample_mean = np.mean(y)
        ssto = np.sum( (y - sample_mean) ** 2)
        r2 = 1. - sse/ssto
        cv_metrics.append({'r2': r2, 'r': r})
    return cv_metrics


def load_params(param_path):
    with open(param_path, 'r') as fo:
        params = json.load(fo)
    return params

if __name__ == '__main__':
    param_path = os.environ['PARAMS']
    pred_path = os.environ['PRED_DIR']
    source_path = os.environ['SOURCE_DIR']

    params = load_params(param_path)

    targets = params["target_cols"]

    for target in targets:
        
        predictions = pd.read_pickle(os.path.join(pred_path, target, "predictions.pkl"))
        features = pd.read_pickle(os.path.join(pred_path, target, "features.pkl"))

        if params['prediction_task'] == 'classification':
            accuracies = calc_accuracy(predictions)
            acc_avg = np.mean([fold["acc"] for fold in accuracies])
            base_avg = np.mean([fold["base"] for fold in accuracies])
            print("ACC: {}".format(acc_avg))
            print("Baseline: {}".format(base_avg))
        elif params['prediction_task'] == 'regression':
            reg_metrics = calc_r2(predictions)
            #rsme = calc_rsme(predictions)
            print("r2: {}".format(np.mean([fold["r2"] for fold in reg_metrics])))
        else:
            print("Invalid prediction_task parameter: {}".format(params["prediction_task"]))

        #print(rsme)
            
