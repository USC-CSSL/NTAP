# NOTE IMPORTANT: Must clone and run on leigh_dev branch

import sys
sys.path.append('.')

from ntap.data import Dataset
from ntap.SVM import SVM
import pandas as pd
import argparse
import os
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input file")
parser.add_argument("--predict", help="Path to predict data")
parser.add_argument("--save", help="Path to save directory")
parser.add_argument("--dictionary", help="Path to input file for DDR or word count")
parser.add_argument("--glove", help="Path to glove embeddings")

args = parser.parse_args()

SEED = 734 # Go gags
random.seed(SEED)

def save_results(res, name, path):
    with open(os.path.join(path, name), 'w') as out:
        res[0].to_csv(out)
    print("Saved results ({}) to {}".format(name, path))

def chunk_data(input_path, chunksize=10000):
    data_iter = pd.read_csv(input_path, chunksize=100000)
    ret_list = []
    for data in data_iter:
        ret_list.append(data)
    return pd.concat(ret_list)
    
def init_model(target, feature, dataset):
    formula = target+" ~ "+feature+"(Text)"
    model = SVM(formula, data=dataset, random_state=SEED)
    return model

def cv(model, data):
    results = model.CV(data=data)
    return results

def train(model, data, params=None):
    model.train(data, params=params)

def process_data(data):
    data.dropna(subset=['body'], inplace=True)
    data = Dataset(data)
    data.clean(column='body')
    return data

def process_dictionary(dictionary, sample_n=10, random_seed=SEED):
    return_dict = {}
    for k, v in dictionary.items():
        if len(v) < sample_n:
            print(k)
        return_dict[k] = random.sample(v, sample_n)
    return return_dict
    
def predict(model, predict_path, feat, filename, params):
    user_all = []
    y_all = []
    text_all = []

    count = 0
    # Chunk so its not read in all at once
    data_iter = pd.read_csv(predict_path, sep='\t', chunksize=100000)
    for data_chunk in data_iter:
        count += 1
        print("Chunk {}".format(count))
        data_chunk = process_data(data_chunk)
        # Get users and text after processing data (rows will be dropped)
        users = data_chunk.data['id']
        text = data_chunk.data['body']
        if feat == "ddr":
            data_chunk.dictionary = params['dictionary']
            data_chunk.glove_path = params['glove_path']
        elif feat == "wordcount":
            data_chunk.dictionary = params['dictionary']
        # Running tfidf/ddr method from Dataset()
        getattr(data_chunk, feat)(column='body')
        y_hat = model.predict(data_chunk)
        y_all.extend(y_hat)
        user_all.extend(users)
        text_all.extend(text)
        chunk_filename = filename + "_"+ str(count)
        # Save over time, just in case it crashes
        if count % 10 == 0:
            pd.DataFrame(list(zip(user_all, text_all, y_all)), columns=["user_id", "text", "y"]).to_csv(filename, index=False)
    pd.DataFrame(list(zip(user_all, text_all, y_all)), columns=["user_id", "text", "y"]).to_csv(filename, index=False)

def evaluate(model, predictions, labels, target):
    stats = model.evaluate(predictions, labels, 2, target)
    return stats

if __name__=='__main__':
    features = ["lda"] # lda, ddr, wordcount
    targets = ["hate", "hd", "vo"] # cv, hd, vo

    input_path = args.input
    output_path = args.save if args.save else os.getcwd()
    params = {}

    if "ddr" in features:
        if not args.dictionary:
            print("No dictionary found for ddr, please specify path to dictionary")
            sys.exit()
        elif not args.glove:
            print("Path to glove embeddings for ddr not found, please specify in glove argument")
            sys.exit()
        params['dictionary'] = process_dictionary(json.load(open(args.dictionary)))
        params['glove_path'] = args.glove
    elif "wordcount" in features:
        if not args.dictionary:
            print("No dictionary found for wordcount, please specify path")
            sys.exit()
        params['dictionary'] = json.load(open(args.dictionary))

    for feat in features:
        for target in targets:
            model_filename = "_".join([target, feat, "cvres.csv"])
            filename = os.path.join(output_path, "_".join([target, feat, "fullgabpred"]))
            data = Dataset(input_path,max_df=0.15)
            if feat == "ddr":
                data.dictionary = params['dictionary']
                data.glove_path = params['glove_path']
            elif feat == "wordcount":
                data.dictionary = params['dictionary']
            model = init_model(target, feat, data)
            cv_res = cv(model, data)
            save_results(cv_res.dfs, model_filename, output_path)
            # print("Training...")
            # train(model, data)
            # print("Predicting...")
            # predict(model, args.predict, feat, filename, params)

