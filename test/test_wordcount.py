import os
from ntap.data import Dataset
from ntap.models import SVM

SEED = 734


def save_results(res, name, path=os.getcwd()):
    with open(os.path.join(path, name), 'w') as out:
        res[0].to_csv(out)
    print("Saved results ({}) to {}".format(name, path))


def run_svm(target, feature, dataset, save_file, save_path):
    formula = target + " ~ " + feature + "(Text)"

    model = SVM(formula, data=dataset, random_state=SEED)
    results = model.CV(data=dataset)
    save_results(results.dfs, save_file, path=save_path)


if __name__ == '__main__':
    data_path = "../data/union_test.tsv"
    dictionary_path = "../data/liwc_2015.json"
    dataset = Dataset(data_path)
    print(dataset.data.head(10))
    dataset.wordcount(column="Text", dictionary=dictionary_path)
    print(dataset.features["wordcount"].shape)

    features = ["wordcount"]  # include lda, ddr, liwc later
    targets = ["hate", "hd", "cv"]

    for feat in features:
        for target in targets:
            filename = target + "_" + feat + ".csv"
            run_svm(target, feat, dataset, filename, "majority_results")
