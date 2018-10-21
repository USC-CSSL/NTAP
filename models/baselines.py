"""
in desparate need of refactoring...
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import LinearSVC

def do_param_grid(grid, X_train, y_train, X_test, y_test, model):
    best_score_ = 0.
    best_grid_ = None
    predictions = None
    for g in ParameterGrid(grid):
        model.set_params(**g)
        model.fit(X_train, y_train)
        score_ = model.score(X_train, y_train)
        if score_ > best_score_:
            best_score = score_
            best_grid_ = g 
            predictions = model.predict(X_test)
    return best_grid_, predictions


class Regressor:
    def __init__(self, params):
        self.method = params["prediction_method"]
        self.kfolds = params["k_folds"]
        self.seed = params["random_seed"]
        self.param_grid = {"alpha": 10.0 ** -np.arange(3, 5),
                           "l1_ratio": np.arange(0.15, 0.25, 0.1)}

    def cv_results(self, X, y):
        self.model = SGDRegressor(loss='squared_loss',  # default
                                  penalty='elasticnet', 
                                  tol=1e-3,
                                  shuffle=True,
                                  random_state=self.seed,
                                  verbose=1)
        kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

        predictions = dict()
        parameters = dict()
        indices = dict()
        features = np.zeros( (X.shape[1], ) )
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            parameter_iter, prediction_iter = do_param_grid(self.param_grid,
                                                X_train, y_train, X_test,
                                                y_test, self.model)
            features += self.model.coef_
            predictions[idx] = prediction_iter
            parameters[idx] = parameter_iter
            indices[idx] = test_idx

        features /= len(predictions)

        return predictions, parameters, indices, features

    def format_results(self, predictions, true_vals, row_indices):
        ### Define a multi-index with (cv-num, doc_idx)
        list_of_cvs = [[i] * len(predictions[i]) for i in list(predictions.keys())]
        cv_nums = [item for sublist in list_of_cvs for item in sublist]
        doc_idxs = [item for sublist in list(row_indices.values()) for item in sublist]
        vals = [item for sublist in list(predictions.values()) for item in sublist]
        #doc_idxs = [item for sublist in list_of_ids for item in sublist]
        arrays = [cv_nums, doc_idxs]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples, names=["CrossVal_fold", "Doc_index"])

        list_of_dicts = [{"y": true_vals[i], "y_hat": vals[i]} for i in range(len(true_vals))]
        return pd.DataFrame(list_of_dicts, index=mindex)

    def format_features(self, features,  feature_names):
        """
        'features' is a (n_classes, n_features) numpy array
        return: dataframe indexed by class, columns by feature_names
        """
        dataframe = pd.DataFrame(features)
        print(dataframe)
        dataframe.index = feature_names

        # sort by the first column, just because
        df = dataframe.assign(f = abs(dataframe[0])).sort_values([0]).drop('f', axis=1)

        return df
class Classifier:
    def __init__(self, params):
        self.method = params["prediction_method"]
        self.kfolds = params["k_folds"]
        self.seed = params["random_seed"]
        self.param_grid = {"class_weight": ['balanced', None],
                           "C": np.arange(0.05, 0.5, 0.05)}

    def cv_results(self, X, y):
        ### k-fold cross-validation: train on 90%, test on 10%; report all results
        # cast target vector to numpy int:
        y_int = y.astype(int)
        # get num-classes:
        num_classes = len(list(set(y_int)))
        if self.method == 'log_regression':
            class_type = 'ovr' if num_classes == 2 else 'multinomial'
            self.model = LogisticRegression(random_state=self.seed, solver='sag', 
                                            multi_class=class_type, verbose=0,
                                            tol=1e-4, max_iter=60)
        elif self.method == 'svm':
            self.model = LinearSVC(dual=False, random_state=self.seed)
        kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

        predictions = dict()
        parameters = dict()
        indices = dict()
        features = np.zeros( ( num_classes, X.shape[1]) )
        for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_int[train_idx], y_int[test_idx]
            parameter_iter, prediction_iter = do_param_grid(self.param_grid, 
                                                            X_train, y_train, 
                                                            X_test, y_test, 
                                                            self.model)
            features += self.model.coef_
            predictions[idx] = prediction_iter
            parameters[idx] = parameter_iter
            indices[idx] = test_idx

        features /= len(predictions)

        return predictions, parameters, indices, features

    def format_results(self, predictions, true_vals, row_indices):
        ### Define a multi-index with (cv-num, doc_idx)
        list_of_cvs = [[i] * len(predictions[i]) for i in list(predictions.keys())]
        cv_nums = [item for sublist in list_of_cvs for item in sublist]
        doc_idxs = [item for sublist in list(row_indices.values()) for item in sublist]
        vals = [item for sublist in list(predictions.values()) for item in sublist]
        #doc_idxs = [item for sublist in list_of_ids for item in sublist]
        arrays = [cv_nums, doc_idxs]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples, names=["CrossVal_fold", "Doc_index"])

        list_of_dicts = [{"y": true_vals[i], "y_hat": vals[i]} for i in range(len(true_vals))]
        return pd.DataFrame(list_of_dicts, index=mindex)

    def format_features(self, features,  feature_names):
        """
        'features' is a (n_classes, n_features) numpy array
        return: dataframe indexed by class, columns by feature_names
        """
        dataframe = pd.DataFrame(features).transpose()
        dataframe.index = feature_names
        dataframe.columns = self.model.classes_

        # sort by the first column, just because
        df = dataframe.assign(f = abs(dataframe[0])).sort_values([0]).drop('f', axis=1)

        return df
