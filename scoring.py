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
    num_folds = len(cv_folds)
    cv_metrics = {"Accuracy": 0.,
                  "Precision": 0.,
                  "Recall": 0.,
                  "F1": 0.}

    for cv_fold in sorted(cv_folds):
        cv_data = predictions[predictions["cv_num"] == cv_fold]
        y = cv_data["y"].values
        y_hat = cv_data["y_hat"].values
        true_pos = np.sum(np.logical_and(y == 1, y_hat == 1))
        true_neg = np.sum(np.logical_and(y == 0, y_hat == 0))
        false_pos = np.sum(np.logical_and(y == 1, y_hat == 0))
        false_neg = np.sum(np.logical_and(y == 0, y_hat == 1))

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        cv_metrics["Recall"] += true_pos / (true_pos + false_neg)
        cv_metrics["Precision"] += true_pos / (true_pos + false_pos)
        cv_metrics["Accuracy"] += (true_pos + true_neg) / len(y)
        cv_metrics["F1"] += 2 * (precision * recall) / (precision + recall)

    for metric in cv_metrics:
        cv_metrics[metric] /= num_folds
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
    parser.add_argument("--task", help="classification or regression")
    args = parser.parse_args()
    if args.predictions.endswith('csv'):
        predictions = pd.read_csv(args.predictions)
    elif args.predictions.endswith('pkl'):
        predictions = pd.read_pickle(args.predictions)
    else:
        raise ValueError("File must be of type .pkl or .csv")

    if args.task == 'classification':
        scores = score_classification(predictions)
        print(json.dumps(scores, indent=4))  
       
    elif args.task == 'regression':  # TODO: Make better
        scores = score_regression(predictions)
        r2_avg = np.mean([fold["r2"] for fold in scores])
        print("R2: {}".format(r2_avg))
    else:
        raise ValueError("Invalid \'task\' parameter: {}".format(args.task))

