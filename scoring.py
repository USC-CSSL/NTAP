"""
file: scoring.py
purpose: given prediction files, generate aggregate performance 
    measures across all folds
"""

import json, os, argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from parameters import prediction as params

def score_classification(predictions):

    cv_folds = set(predictions["cv_num"].values)
    cv_metrics = list()

    for cv_fold in sorted(cv_folds):
        cv_data = predictions[predictions["cv_num"] == cv_fold]
        num_correct, base_correct = 0, 0
        scores = list()
        for mess_id, row in cv_data.iterrows():
            if row['y'] == row['y_hat']:
                num_correct += 1
        perf = ( 1. * num_correct) / len(cv_data)
        cv_metrics.append({"acc": perf})
    return cv_metrics


def score_regression(predictions):

    cv_folds = set(predictions["cv_num"].values)
    cv_metrics = list()

    for cv_fold in sorted(cv_folds):
        cv_data = predictions[predictions["cv_num"] == cv_fold]
        y = cv_data['y'].values
        y_hat = cv_data['y_hat'].values

        r = pearsonr(y, y_hat)

        sse = np.sum( (y - y_hat) ** 2)
        sample_mean = np.mean(y)
        ssto = np.sum( (y - sample_mean) ** 2)
        r2 = 1. - sse/ssto
        cv_metrics.append({'r2': r2, 'r': r})
    return cv_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="Path to predictions dataframe")
    args = parser.parse_args()
    task = params["prediction_task"] 
    predictions = pd.read_pickle(args.predictions)

    if task == 'classification':
        scores = score_classification(predictions)
        acc_avg = np.mean([fold["acc"] for fold in scores])
        print("ACC: {}".format(acc_avg))
    elif task == 'regression':
        scores = score_regression(predictions)
        print("r2: {}".format(np.mean([fold["r2"] for fold in scores])))
    else:
        raise ValueError("Invalid \'task\' parameter: {}".format(task))

