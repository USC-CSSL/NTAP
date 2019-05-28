#TODO: Add CNN and RNN models to files cnns.py and rnns.py
from helperFunctions import *
from methods.neural.neural import Neural
import pandas as pd
import pickle
import argparse
import numpy as np
from collections import Counter
import json


"""
def load_method(method_string):
    method_map = {"CNN": CNN}  # TODO: Add others
    if method_string not in method_map:
        raise ValueError("Invalid method string ({})".format(method_string))
        exit(1)
    return method_map[method_string]
"""
class Methods:
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


    def run_method(self, all_params, train_data, data, save, features):
        missing_indices = list()
        weights = dict()
        method_string = all_params['model']['method']
        neural_params = all_params['neural_params']
        # TODO: Add exception handling
        text_col = "text"#__get_text_col(train_data.columns.tolist())
        #params["target_cols"] = self.__get_target_col(train_data.columns.tolist())
        #params["pred_cols"] = __get_pred_col(train_data.columns.tolist())
        for target in neural_params["target_cols"]:
            print("Removing missing values from", target, "column")
            missing_indices.extend(train_data[train_data[target] == -1.].index)
            print("Statistics of", target)
            count_pre = Counter(train_data[target].tolist())
            count = sorted(count_pre.items())
            count = list(dict(count).values())
            weights[target] = np.array([(sum(count) - c) / sum(count) for c in count])
        train_data = train_data.drop(missing_indices)
        train_data = train_data.dropna(subset=[text_col])
        print("Shape of train_dataframe after getting rid of the Nan values is", train_data.shape)

        train_data = tokenize_data(train_data, text_col, neural_params["max_length"], neural_params["min_length"])
        text = train_data[text_col].values.tolist()

        print(len(text), "text data remains in the dataset")
        vocab = learnVocab(text, neural_params["vocab_size"])
        # vocab_size = len(vocab)
        # method_class = load_method(method_string)

        X = np.array(tokens_to_ids(text, vocab))
        y = np.transpose(np.array([np.array(train_data[target].astype(int)) for target in neural_params["target_cols"]]))
        # y = np.transpose(np.array(np.array(train_data[neural_params["target_cols"]].astype(int))))
        if neural_params["model"][-4:] == "feat":
            if features.endswith('.tsv'):
                feat = pd.read_csv(features, sep='\t', quoting=3).values
            elif features.endswith('.pkl'):
                feat = pickle.load(open(features, 'rb')).values
            elif features.endswith('.csv'):
                feat1  = pd.read_csv(features)
                feat = feat1.loc[:,["care", "harm", "fairness", "cheating"]].values
            neural_params["feature_size"] = feat.shape[1]
        else:
            feat = []
        neural = Neural(all_params, vocab)
        neural.build()

        if neural_params["train"]:
            if neural_params["kfolds"]<=1:
                raise Exception('Please set the parameter, kfolds greater than 1')
            neural.trainModelUsingCV(X, y, weights, save, feat)

        if neural_params["predict"]:
            data = all_params["processing"]["input_path"]
            if data is None:
                raise Exception("Please specify the path to the data to be predicted")
            if data.endswith('.tsv'):
                all = pd.read_csv(data, sep='\t', quoting=3)
            elif data.endswith('.pkl'):
                all = pickle.load(open(data, 'rb'))
            #elif data.endswith('.csv'):
            else:
                all = pd.read_csv(data)

            #col = self.__get_text_col(all.columns.tolist())
            col = "text"
            all = tokenize_data(all, col, neural_params["max_length"], neural_params["min_length"])
            all_text = all[col].values.tolist()
            data = np.array(tokens_to_ids(all_text, vocab))

            neural.predictModel(X, y, data, weights, save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Path to train data; includes text and any additional \
                        train_data columns, such as POS")
    parser.add_argument("--data", help="Path to data to be labeled")
    parser.add_argument("--method", help="Method string; see README for complete list")
    parser.add_argument("--params", help="Path to parameter file; can be .txt, .json, .csv")
    parser.add_argument("--savedir", help="Directory to save results in")
    parser.add_argument("--feature", help="Path to the features file")

    args = parser.parse_args()

    if args.train_data.endswith('.tsv'):
        train_data = pd.read_csv(args.train_data, sep='\t', quoting=3)
    elif args.train_data.endswith('.pkl'):
        train_data = pd.read_pickle(args.train_data)
    elif args.train_data.endswith('.csv'):
        train_data = pd.read_csv(args.train_data)

    with open('params.json') as f:
        params = json.load(f)

    data = args.data
    method = args.method
    params = params['neural_params']
    features = args.feature
    save = args.savedir
    """
    params = args.params
    with open(params, 'r') as fo:
        if params.endswith(".txt"):
            params = params.read()
        elif params.endswith(".json"):
            params = json.load(fo)
        elif params.endswith(".csv"):
            params = pd.read_csv(params)
        else:
            raise ValueError("Error: Incorrect extension for parameter file, must be a .txt, .json, or a .csv")
    """
    method = Methods()
    method.run_method(method, train_data, params, data, save, features)

    # call method constructors; build, train (using columns in args.train_data), and generate predictions
