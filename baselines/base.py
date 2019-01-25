import os
from sklearn.model_selection import KFold, GridSearchCV
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

seed = 123

class BasePredictor(ABC):
    """
    Base unit for all other predictor classes
    Facilitates formatting of results, cross-validation, 
        and grid search
    """
    def __init__(self, dest, row_ids, k_folds):
        self.dest = dest
        self.row_ids = row_ids
        if not os.path.isdir(self.dest):
            os.makedirs(self.dest)
        self.k_folds = k_folds

    @abstractmethod
    def build(self):
        pass

    def train(self, X, y):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
        # just write as I go
        results_path = os.path.join(self.dest, "predictions.csv")
        out = open(results_path, 'w')

        self.pred = dict()
        self.targets = dict()
        self.features = dict()
        self.indices = dict()
        columns = ["row_idx", "cv_num", "y", "y_hat"]
        out.write(','.join(columns))
        out.write('\n')
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            estimator = GridSearchCV(self.model, self.param_grid, cv=3)
            estimator.fit(X=X_train, y=y_train)
            pred = estimator.predict(X_test)

            self.features[idx] = estimator.best_estimator_.coef_
            
            num_rows = len(pred)
            ids = [self.row_ids[idx] for idx in test_idx]
            rows = zip(ids, [idx] * num_rows, y_test, pred)
            out.write('\n'.join([','.join([str(val) for val in row]) for row in rows]))
            out.flush()
        out.close()
    @abstractmethod
    def format_features(self, feature_names):
        pass

