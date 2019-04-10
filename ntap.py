import fnmatch

from process.processor import Preprocessor
import json
import os
from baselines.features import Features
from baselines.methods import Baseline

from methods.neural.nn import *
from methods.neural.neural import Neural
import pandas as pd
import pickle
import numpy as np
from collections import Counter

class Ntap:

    def __init__(self, params):
        self.params = params
        self.base_dir = os.path.dirname(params['processing']['input_path'])
        self.pre_filename = "data"
        self.pre_ftype = ".pkl"
        self.preprocessed_dir = os.path.join(self.base_dir, "preprocessed")
        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        self.feature_dir = os.path.join(self.base_dir, "features")
        self.preprocessed_file = os.path.join(self.preprocessed_dir, self.pre_filename + self.pre_ftype)
        self.data = None
        self.test_filepath = params['model']['test_filepath']
        self.model_path = os.path.join(self.base_dir, "models")
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def load_preprocessed_data(self, file):
        if file.endswith('.tsv'):
            target = pd.read_csv(file, sep='\t', quoting=3)
        elif file.endswith('.pkl'):
            target = pd.read_pickle(file)
        elif file.endswith('.csv'):
            target = pd.read_csv(file)
        return target

    def preprocess(self, params):
        jobs = params['processing']['jobs']

        processor = Preprocessor(self.preprocessed_dir, self.params)
        try:
            processor.load(self.preprocessed_file)
        except Exception as e:
            print(e)
            print("Could not load data from {}".format(self.base_dir))
            exit(1)

        for job in jobs:
            print("Processing job: {}".format(job))
            if job == 'clean':
                processor.clean(params["processing"]["clean"], remove=True)

            if job == 'ner':
                processor.ner()
            if job == 'pos':
                processor.pos()
            if job == 'depparse':
                processor.depparse()
            if job == 'tagme':
                processor.tagme()
        processor.write(self.pre_ftype)
        self.data = processor.data

    def baseline(self):
        feature_list = self.params['baseline']['features']
        if feature_list:
            feature_to_fit = []
            for feat_str in feature_list:
                feat_files = fnmatch.filter(os.listdir(self.feature_dir), feat_str + '.*')
                if not feat_files:
                    feature_to_fit.append(feat_str)
            if feature_to_fit:
                feature_pipeline = Features(self.base_dir, self.params)
                feature_pipeline.load(self.preprocessed_file)
                for feat_str in feature_to_fit:
                    feature_pipeline.fit(feat_str)
                    feature_pipeline.transform()  # writes to file
        method = self.params['baseline']['method']
        if method:
            baseline_pipeline = Baseline(self.base_dir, self.params)
            targets = self.params['baseline']['targets']
            if not targets:
                baseline_pipeline.load_data(self.preprocessed_file)
            else:
                baseline_pipeline.load_data(self.preprocessed_file, targets)
            baseline_pipeline.load_features()
            baseline_pipeline.load_method(method)
            baseline_pipeline.go()

    def __get_text_col(self, cols):
        print("...".join(cols))
        notvalid = True
        while notvalid:
            text_col = input("Enter text col from those above: ")
            if text_col.strip() not in cols:
                print("Not a valid column name")
            else:
                notvalid = False
        return text_col

    def __get_target_col(self, cols):
        notvalid = True
        while notvalid:
            target_col = input("Enter target cols you want to train the model on, from those above: ")
            targets = target_col.strip().replace(" ", "").split(",")
            notvalid = False
            for target in targets:
                if target not in cols:
                    notvalid = True
                    print("Not a valid column name")
        return targets

    def __get_pred_col(self, cols):
        notvalid = True
        while notvalid:
            target_col = input("Enter target cols you want to predict, from those above: ")
            targets = target_col.strip().replace(" ", "").split(",")
            notvalid = False
            for target in targets:
                if target not in cols:
                    notvalid = True
                    print("Not a valid column name")
        return targets

    def run_method(self, params):
        missing_indices = list()
        weights = dict()
        neural_params = params['neural_params']
        # TODO: Add exception handling
        text_col = "text"  # __get_text_col(train_data.columns.tolist())
        neural_params["target_cols"] = self.__get_target_col(self.data.columns.tolist())
        # params["pred_cols"] = __get_pred_col(train_data.columns.tolist())
        for target in neural_params["target_cols"]:
            print("Removing missing values from", target, "column")
            missing_indices.extend(self.data[self.data[target] == -1.].index)
            print("Statistics of", target)
            count_pre = Counter(self.data[target].tolist())
            count = sorted(count_pre.items())
            count = list(dict(count).values())
            weights[target] = np.array([(sum(count) - c) / sum(count) for c in count])
        train_data = self.data.drop(missing_indices)
        train_data = train_data.dropna(subset=[text_col])
        print("Shape of train_dataframe after getting rid of the Nan values is", train_data.shape)

        train_data = tokenize_data(train_data, text_col, neural_params["max_length"], neural_params["min_length"])
        text = train_data[text_col].values.tolist()

        print(len(text), "text data remains in the dataset")
        vocab = learn_vocab(text, neural_params["vocab_size"])
        # vocab_size = len(vocab)
        # method_class = load_method(method_string)

        X = np.array(tokens_to_ids(text, vocab))
        y = np.transpose(np.array([np.array(train_data[target].astype(int)) for target in neural_params["target_cols"]]))
        # y = np.transpose(np.array(np.array(train_data[params["target_cols"]].astype(int))))
        feature = params['model']['feature']
        feature_file = fnmatch.filter(os.listdir(self.feature_dir), feature + '.*')[0]
        feature_path = os.path.join(self.feature_dir, feature_file)
        if neural_params["model"][-4:] == "feat":
            if feature_path.endswith('.tsv'):
                feat = pd.read_csv(feature_path, sep='\t', quoting=3).values
            elif feature_path.endswith('.pkl'):
                feat = pickle.load(open(feature_path, 'rb')).values
            elif feature_path.endswith('.csv'):
                feat1 = pd.read_csv(feature_path)
                feat = feat1.loc[:, ["care", "harm", "fairness", "cheating"]].values
            neural_params["feature_size"] = feat.shape[1]
        else:
            feat = []
        neural = Neural(params, vocab)
        neural.build()

        if neural_params["train"]:
            if neural_params["kfolds"] <= 1:
                raise Exception('Please set the parameter, kfolds greater than 1')
            neural.cv_model(X, y, weights, self.model_path, feat)

        if neural_params["predict"]:
            if self.test_filepath is None:
                raise Exception("Please specify the path to the data to be predicted")
            all = self.load_preprocessed_data(self.test_filepath)

            col = self.__get_text_col(all.columns.tolist())
            all = tokenize_data(all, col, neural_params["max_length"], neural_params["min_length"])
            all_text = all[col].values.tolist()
            data = np.array(tokens_to_ids(all_text, vocab))

            neural.run_model(X, y, data, weights, self.model_path)


if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    ntap = Ntap(params)
    if not os.listdir(ntap.preprocessed_dir):
        ntap.preprocess(params)
    else:
        ntap.data = ntap.load_preprocessed_data(ntap.preprocessed_file)
    ntap.baseline()
    ntap.run_method(params)
