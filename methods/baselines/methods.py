
import pandas as pd
import numpy as np
import os, json

from sklearn.linear_model import SGDRegressor, Lasso
from sklearn.svm import LinearSVC

from methods.baselines.base import BasePredictor

seed = 123

class ElasticNet(BasePredictor):
    def __init__(self, save_dir, instance_names,
                 k_folds, param_grid=None):
        BasePredictor.__init__(self, save_dir, instance_names, k_folds)
        if param_grid is None:  # default
            self.param_grid = {}  #"alpha": 10.0 ** -np.arange(3, 7)}
        else:
            self.param_grid = param_grid
        
    
    def build(self):
        self.model = SGDRegressor(loss='squared_loss',
                                  penalty='none', #'elasticnet', Just for DDR
                                  tol=1e-3, # changed cuz it takes a long time
                                  shuffle=True,
                                  random_state=seed,
                                  verbose=10)
                              
        # Note: Lasso doesn't do too well/different than ElasticNet
        # Dev TODO: Modularize regression model selection
        # Dev TODO: Standardize inputs and targets, optionally. Decide where...

    def format_features(self, feature_names):
       
        num_features = len(feature_names)
        features = np.zeros( (num_features, ) )
        for _, coef in self.features.items():
            features += coef
        features /= self.k_folds

        dataframe = pd.DataFrame(features)
        dataframe.index = feature_names

        return dataframe

class SVM(BasePredictor):
    def __init__(self, save_dir, instance_names, k_folds, 
                       n_classes=2, param_grid=None):
        BasePredictor.__init__(self, save_dir, instance_names, k_folds)
        self.k_folds = k_folds
        self.n_classes = n_classes
        if param_grid is None:  # default
            self.param_grid = {"class_weight": ['balanced'],
                               "C": [1.0]}  #np.arange(0.05, 1.0, 0.05)}
        else:
            self.param_grid = param_grid

    def build(self):
        self.model = LinearSVC(C=1.0, class_weight='balanced', 
                               dual=False, random_state=seed)
    
    def format_features(self, feature_names):
        # self.features is a dict of matrices (num_features) or (n_classes, num_features)
        num_features = len(feature_names)
        if self.n_classes > 2:
            features = np.zeros( (self.n_classes, num_features) )
        else:
            features = np.zeros( (num_features) )
        for _, coef in self.features.items():
            if self.n_classes == 2:
                coef = coef.reshape( (num_features,) )
            features += coef
        
        if self.n_classes > 2:
            f = pd.DataFrame(features).transpose()
            f.index = feature_names
            f.columns = self.model.classes_
        else:
            f = pd.Series(features, index=feature_names)
        return f

class Baseline:
    def __init__(self, dir_, params):
        self.feat_dir = os.path.join(dir_, "features")
        if not os.path.isdir(self.feat_dir):
            raise ValueError("Cannot read features from {}".format(self.feat_dir))
        self.dest_dir = os.path.join(dir_, "models")
        self.params = params['feature_params']
        if not os.path.isdir(self.dest_dir):
            os.makedirs(self.dest_dir)
    
    def __load_file(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.pkl':
            return pd.read_pickle(path)
        elif ext == '.csv':
            return pd.read_csv(path, index_col=0)
        elif ext == '.tsv':
            return pd.read_csv(path, delimiter='\t', index_col=0)

    def __get_target_cols(self, cols):
        print("...".join(cols))
        notvalid = True
        valid_cols = list()
        while notvalid:
            target_str = input("Enter target_columns from those above. \nFor multiple targets, separate them by a comma: ")
            target_cols = target_str.strip().split(',')
            for col in target_cols:
                if col not in cols:
                    print("Not a valid column name: {}".format(col))
                else:
                    valid_cols.append(col)
            if len(valid_cols) > 0:
                print("Using cols: {}".format(",".join(valid_cols)))
                notvalid = False
        return target_cols
    # loads dataframe including target variable(s)
    def load_data(self, data_path, target_cols=None):
        try:
            self.data = self.__load_file(data_path)
        except Exception:
            print("Invalid data file: {}".format(data_path))
            return
        cols = list(self.data.columns)
        if target_cols is not None:
            self.targets = target_cols
        else:
            self.targets = self.__get_target_cols(cols)

    def load_features(self):
        feature_dfs = list()
        for f in os.listdir(self.feat_dir):
            if not os.path.isfile(os.path.join(self.feat_dir, f)):
                continue  # skip directories
            add_features = True
            get_input = True
            while get_input:
                a = input("Load features from {}? (yes/no)".format(f))
                if a == "no":
                    add_features = False
                    get_input = False
                elif a == "yes":
                    get_input = False
            if add_features:
                try:
                    featpath = os.path.join(self.feat_dir, f)
                    data = self.__load_file(featpath)
                    feature_dfs.append(data)
                except Exception:
                    print("Could not load file from {}".format(self.feature_dir))
        if len(feature_dfs) == 0:
            raise ValueError("Couldn't load features from {}".format(self.feat_dir))
        self.features = pd.concat(feature_dfs, axis=1)
    
    def print_features(self):
        print(self.features)
    
    def load_method(self, method_string):
        
        method_map = {"svm": SVM,
                      "elasticnet": ElasticNet} 
        if method_string not in method_map:
            raise ValueError("Invalid method string ({})".format(method_string))
            exit(1)
        if method_string == 'svm':
            self.task = 'classification'
        elif method_string == 'elasticnet':
            self.task = 'regression'
        self.method = method_map[method_string]

    def __write_features(self, res_dir, model_feats):
        model_feats.index.name = "feature_name"
        model_feats.columns = ["feature_weight"]
        model_feats.to_csv(os.path.join(res_dir, "model_weights.csv")) 

    def go(self):
        X = self.features.values
        for target in self.targets:
            # make results directory (pass to method class)
            res_dir = os.path.join(self.dest_dir, target)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            y = self.data[target].values
            if self.task == 'classification':
                n_classes = len(set(y.tolist()))
                y = y.astype(np.int32)
            instance_names = list(self.features.index)

            if self.task == 'classification':
                method_obj = self.method(res_dir, instance_names,
                                    self.params["kfolds"], n_classes)
            else:
                method_obj = self.method(res_dir, instance_names,
                                         self.params["kfolds"])
            method_obj.build()
            method_obj.train(X, y)
            feature_names = self.features.columns.tolist()
            model_features = method_obj.format_features(feature_names)
            self.__write_features(res_dir, model_features) 
