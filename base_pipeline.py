import os, json
import pandas as pd
import numpy as np
import argparse

from baselines.methods import Baseline
from baselines.features import Features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--targets", help="Column names of target variables", 
                        nargs='*')
    parser.add_argument("--features", help="List of feature sets", nargs='*')
    parser.add_argument("--method", help="Method string (svm/elasticnet)")
    parser.add_argument("--destdir", help="Path to save directory")
    parser.add_argument("--config", help="Path to config file (json/txt/csv)")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as fo:
            params = json.load(fo)
    except Exception as e:
        print("Bad parameter file: {}".format(args.config))

    if args.features is not None:
        feature_pipeline = Features(args.destdir, params)
        feature_pipeline.load(args.input)
        for feat_str in args.features:
            feature_pipeline.fit(feat_str)
            feature_pipeline.transform()  # writes to file

    if args.method is not None:
        baseline_pipeline = Baseline(args.destdir, params)
        if args.targets is None:
            baseline_pipeline.load_data(args.input)
        else:
            baseline_pipeline.load_data(args.input, args.targets)
        baseline_pipeline.load_features()
        baseline_pipeline.load_method(args.method)
        baseline_pipeline.go()
