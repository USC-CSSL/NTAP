import sys
sys.path.append('.')

from ntap.data import Dataset
from ntap.models import SVM
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to input file")
parser.add_argument("--output", help="Path to output directory")

args = parser.parse_args()

SEED = 734 # Go Blue!
# BEST_SCORE = 0
# BEST_MODEL = []

def save_results(res, name, path=os.getcwd()):
    with open(os.path.join(path, name), 'w') as out:
        res[0].to_csv(out)
    print("Saved results ({}) to {}".format(name, path))


def run_svm(target, feature, dataset):
    formula = target+" ~ "+feature+"(Text)"
    model = SVM(formula, data=dataset, random_state=SEED)
    results = model.CV(data=dataset)
    return vars(model), results

def predict(model, train_X, params):
    pred = LinearSVC(**params._asdict())

    return model.predict(train_X)

# 
# def get_best_model(model, model_results, metric=""):
#     model_results = model_results[0]
#     if model_results[metric].mean() > BEST_SCORE:
#        BEST_SCORE = model_results[metric].mean()
#        BEST_MODEL = model


if __name__=='__main__':
    features = ["tfidf"] # include lda, ddr, liwc later
    targets = ["hate", "cv", "hd"]

    input_path = args.input
    output_path = args.output if args.output else ""

    for feat in features:
        for target in targets:
            data = Dataset(input_path)
            filename = target + "_" + feat + ".csv"
            model, results = run_svm(target, feat, data)
            print(model)
            print(results.dfs)
            # save_results(results.dfs, filename, output_path)

