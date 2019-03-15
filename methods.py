#TODO: Add CNN and RNN models to files cnns.py and rnns.py
from methods.neural.nn import *
from methods.neural.neural import Neural
from parameters import neural as neural_params
import pandas as pd
import pickle
import argparse
import numpy as np
from collections import Counter


"""
def load_method(method_string):
    method_map = {"CNN": CNN}  # TODO: Add others
    if method_string not in method_map:
        raise ValueError("Invalid method string ({})".format(method_string))
        exit(1)
    return method_map[method_string]
"""

def __get_text_col(cols):
    print("...".join(cols))
    notvalid = True
    while notvalid:
        text_col = input("Enter text col from those above: ")
        if text_col.strip() not in cols:
            print("Not a valid column name")
        else:
            notvalid = False
    return text_col


def __get_target_col(cols):
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

def __get_pred_col(cols):
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


def run_method(method_string, train_data, params, data, save, features):
    missing_indices = list()
    weights = dict()

    # TODO: Add exception handling
    text_col = __get_text_col(train_data.columns.tolist())
    params["target_cols"] = __get_target_col(train_data.columns.tolist())
    #params["pred_cols"] = __get_pred_col(train_data.columns.tolist())
    for target in params["target_cols"]:
        print("Removing missing values from", target, "column")
        missing_indices.extend(train_data[train_data[target] == -1.].index)
        print("Statistics of", target)
        count = Counter(train_data[target].tolist())
        count = list(dict(count).values())
        print(count)
        weights[target] = np.array([(sum(count) - c) / sum(count) for c in count])
    train_data = train_data.drop(missing_indices)
    print("Shape of train_dataframe after getting rid of the Nan values is", train_data.shape)

    text = train_data[text_col].values.tolist()
    text = tokenize_data(text, params["max_length"])
    vocab = learn_vocab(text, params["vocab_size"])
    # vocab_size = len(vocab)
    # method_class = load_method(method_string)

    X = np.array(tokens_to_ids(text, vocab))
    y = np.transpose(np.array([np.array(train_data[target].astype(int)) for target in params["target_cols"]]))
    # y = np.transpose(np.array(np.array(train_data[params["target_cols"]].astype(int))))
    if params["model"][-4:] == "feat":
        if features.endswith('.tsv'):
            feat = pd.read_csv(features, sep='\t', quoting=3).values
        elif features.endswith('.pkl'):
            feat = pickle.load(open(features, 'rb')).values
        elif features.endswith('.csv'):
            feat  = pd.read_csv(features).values
        params["feature_size"] = feat.shape[1]
    else:
        feat = []
    neural = Neural(params, vocab)
    neural.build()

    if params["train"]:
        if params["kfolds"]<=1:
            raise Exception('Please set the parameter, kfolds greater than 1')
        neural.cv_model(X, y, weights, save, feat)

    if params["predict"]:
        if data is None:
            raise Exception("Please specify the path to the data to be predicted")
        if data.endswith('.tsv'):
            all = pd.read_csv(data, sep='\t', quoting=3)
        elif data.endswith('.pkl'):
            all = pickle.load(open(data, 'rb'))
        elif data.endswith('.csv'):
            all = pd.read_csv(data)

        all_text = all[__get_text_col(all.columns.tolist())].values.tolist()
        all_text = tokenize_data(all_text, params["max_length"])
        data = np.array(tokens_to_ids(all_text, vocab))

        neural.run_model(X, y, data, weights, save)

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
        train_data = pickle.load(open(args.train_data, 'rb'))
    elif args.train_data.endswith('.csv'):
        train_data = pd.read_csv(args.train_data)


    data = args.data
    method = args.method
    params = neural_params
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
    run_method(method, train_data, params, data, save, features)

    # call method constructors; build, train (using columns in args.train_data), and generate predictions

