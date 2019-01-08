import os, json
import pandas as pd
import numpy as np
import argparse

from methods.baselines import SVM, ElasticNet
#TODO: Add CNN and RNN models to files cnns.py and rnns.py
#from methods.cnns import *
#from methods.rnns import *

from parameters import prediction as pred_params
from parameters import neural as neural_params

def load_method(method_string):
    method_map = {"svm": SVM,
                  "elasticnet": ElasticNet}  # more to come
    if method_string not in method_map:
        raise ValueError("Invalid method string ({})".format(method_string))
        exit(1)

    #TODO: add initialization params to load_method

    return method_map[method_string]

def baseline(method_string, data, features, pred_task):
    
    X = features.values
    y = data["target"].values
    if pred_task == 'classification':
        n_classes = len(set(y.tolist()))
        y = y.astype(np.int32)
    feature_names = features.columns.tolist()
    instance_names = list(features.index)

    method_class = load_method(method_string)

    if pred_task == 'classification':
        method_obj = method_class(pred_params["kfolds"], n_classes)
    else:
        method_obj = method_class(pred_params["kfolds"])
    method_obj.build()
    method_obj.train(X, y)
    results = method_obj.format_results(instance_names)
    top_features = method_obj.format_features(feature_names)

    return results, top_features
    

def entitylinking(method_string, data, features):
    print("NOT IMPLEMENTED: ENTITY LINKING")
    print("Entity linking: Use features from entity linking; and or text \
           from data, to train and predict for entity linking methods")

def dnn(method_string, data):

    docs = data["text"].values.tolist()
    target = data["target"].values

    method_obj = load_method(method_string)
    print("Use {} to train and predict".format(method_string))
    #TODO: Complete code to run neural methods


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Full dataframe path")
    parser.add_argument("--features", help="Feature dataframe path", default=None)
    parser.add_argument("--results", help="Results dataframe path")
    parser.add_argument("--topfeatures", help="Top features path")
    args = parser.parse_args()

    data = pd.read_pickle(args.data)
    pred_task = pred_params["prediction_task"]
    method_type = pred_params["method_type"]
    method_string = pred_params["method_string"]

    if method_type == 'baseline':
        # method uses features (svm, elasticnet)
        features = pd.read_pickle(args.features)
        results, topfeatures = baseline(method_string, data, features, pred_task)
        results.to_pickle(args.results)
        topfeatures.to_pickle(args.topfeatures)
    elif method_type == 'entitylinking':
        features = pd.read_pickle(args.features)
        entitylinking(method_string, data, features)
    elif method_type == 'dnn':  # deep neural net
        # no features here
        dnn(method_string, data)
    
