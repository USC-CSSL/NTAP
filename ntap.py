import fnmatch
import json
import os
import pandas as pd
from process.processor import Preprocessor
from baselines.methods import Baseline
from baselines.features import Features
from run_methods import Methods


class Ntap:

    def __init__(self, params):
        self.params = params
        self.base_dir,self.filename = os.path.split(params['processing']['input_path'])

        self.preprocessed_dir = os.path.join(self.base_dir, "preprocessed")
        self.feature_dir = os.path.join(self.base_dir, "features")
        self.filetype="."+ self.filename.split(".")[1]
        if not os.path.isdir(self.feature_dir):
            os.makedirs(self.feature_dir)
        self.preprocessed_file = os.path.join(self.preprocessed_dir, self.filename )
        self.input_file = os.path.join(self.base_dir, self.filename )
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
            processor.load(self.input_file)
        except Exception as e:
            print(e)
            print("Could not load data from 1 {}".format(self.base_dir))
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
        processor.write(self.filetype)
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

    def run(self):
        method = Methods()
        feature_file = os.path.join(self.feature_dir, params['model']['feature'] + '.tsv')
        method.run_method(params['model']['method'], self.data, params['neural_params'], self.test_filepath,
                          self.model_path, feature_file)

if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    ntap = Ntap(params)

    if not os.path.isdir(ntap.preprocessed_dir):
         os.makedirs(ntap.preprocessed_dir)
         ntap.preprocess(params)
    else:
         ntap.data = ntap.load_preprocessed_data(ntap.preprocessed_file)
    ntap.baseline()
    ntap.run()
