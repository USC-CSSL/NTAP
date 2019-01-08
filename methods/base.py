import os

from sklearn.model_selection import KFold, GridSearchCV
from abc import ABC, abstractmethod
import pandas as pd

seed = int(os.environ["random_seed"])

class BasePredictor(ABC):
    """
    Base unit for all other predictor classes
    Facilitates formatting of results, cross-validation, 
        and grid search
    """
    def __init__(self, k_folds):
        self.k_folds = k_folds

    @abstractmethod
    def build(self):
        pass

    def train(self, X, y):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)

        self.pred = dict()
        self.targets = dict()
        self.features = dict()
        self.indices = dict()
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            estimator = GridSearchCV(self.model, self.param_grid)
            estimator.fit(X=X_train, y=y_train)
            pred = estimator.predict(X_test)

            self.features[idx] = estimator.best_estimator_.coef_
            self.pred[idx] = pred
            self.targets[idx] = y_test
            self.indices[idx] = test_idx

    def format_results(self, index):
        data = pd.DataFrame(columns=["cv_num", "y", "y_hat", "row_idx"])
        for kf_idx in self.pred:
            for instance_idx in range(len(self.pred[kf_idx])):
                y_hat = self.pred[kf_idx][instance_idx]
                y = self.targets[kf_idx][instance_idx]
                row_idx = index[ self.indices[kf_idx][instance_idx] ]
                data = data.append({"cv_num": kf_idx, 
                                    "y": y, 
                                    "y_hat": y_hat, 
                                    "row_idx": row_idx}, ignore_index=True)
        
        return data

    @abstractmethod
    def format_features(self, feature_names):
        pass

"""
class BaseRNN(ABC)
...
class BaseCNN(ABC)
"""
