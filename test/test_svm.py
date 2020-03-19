# NOTE IMPORTANT: Must clone and run on leigh_dev branch

import sys
sys.path.append('.')

from ntap.data import Dataset
from ntap.models import SVM
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to input file")
parser.add_argument("--predict", help="Path to predict data")
parser.add_argument("--save", help="Path to save directory")

args = parser.parse_args()

SEED = 734 # Go Blue!

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

def predict(model, predict_path, feat, filename):
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
            data_chunk.dictionary="../../HateAnnotations/mfd2.json"
            data_chunk.glove_path = "../../embeddings/glove.6B.300d.txt"
        # Running tfidf/ddr method from Dataset()
        getattr(data_chunk, feat)(column='body')
        y_hat = model.predict(data_chunk)
        y_all.extend(y_hat)
        user_all.extend(users)
        text_all.extend(text)
        chunk_filename = filename + "_"+ str(count)
        # Save over time, just in case it crashes
        if count % 10 == 0:
            pd.DataFrame(list(zip(user_all, text_all, y_all)), columns=["user_id", "text", "y"]).to_csv(chunk_filename, index=False)
            pd.DataFrame(list(zip(user_all, text_all, y_all)), columns=["user_id", "text", "y"]).to_csv(filename, index=False)
    return zip(user_all, y_all, text_all)

def evaluate(model, predictions, labels, target):
    stats = model.evaluate(predictions, labels, 2, target)
    return stats

if __name__=='__main__':
    features = ["ddr"] # lda, ddr, liwc
    targets = ["hate", "cv", "hd", "vo"] # cv, hd, vo

    input_path = args.input
    output_path = args.save if args.save else os.getcwd()

    for feat in features:
        for target in targets:
            model_filename = os.path.join(output_path, "_".join([target, feat, "cv_model"]))
            filename = os.path.join(output_path, "_".join([target, feat, "fullgabpred"]))
            data = Dataset(input_path)
            if feat == "ddr":
                data.dictionary="../../HateAnnotations/mfd2.json"
                data.glove_path = "../../embeddings/glove.6B.300d.txt"
            model = init_model(target, feat, data)
            cv_res = cv(model, data)
            save_results(cv_res.dfs, model_filename, output_path)
            print("Training...")
            train(model, data)
            print("Predicting...")
            results = predict(model, args.predict, feat, filename)
            pd.DataFrame(list(results), columns=["user_id", "y", "text"]).to_csv(filename, index=False)

