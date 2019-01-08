
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVC

from methods.base import BasePredictor

seed = int(os.environ["random_seed"])

class ElasticNet(BasePredictor):
    def __init__(self, k_folds, param_grid=None):
        BasePredictor.__init__(self, k_folds)
        if param_grid is None:  # default
            {"alpha": 10.0 ** -np.arange(3, 5),
             "l1_ratio": np.arange(0.15, 0.45, 0.05)}
        else:
            self.param_grid = param_grid
        
    
    def build(self):
        self.model = SGDRegressor(loss='squared_loss',
                                  penalty='elasticnet', 
                                  tol=1e-3,
                                  shuffle=True,
                                  random_state=seed,
                                  verbose=0)
    
    def format_features(self, feature_names):
       
        num_features = len(feature_names)
        features = np.zeros( (num_features, ) )
        for _, coef in self.features.items():
            features += coef
        features /= self.k_folds

        dataframe = pd.DataFrame(features).transpose()
        dataframe.index = feature_names

        return df

class SVM(BasePredictor):
    def __init__(self, k_folds, n_classes=2, param_grid=None):
        self.k_folds = k_folds
        self.n_classes = n_classes
        if param_grid is None:  # default
            self.param_grid = {"class_weight": ['balanced', None],
                               "C": np.arange(0.05, 1.0, 0.05)}
        else:
            self.param_grid = param_grid

    def build(self):
        self.model = LinearSVC(C=1.0, class_weight=None, 
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
