import os
import abc
import logging
from typing import Optional, Union

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from scipy import sparse
#import optuna
#optuna.logging.set_verbosity(optuna.logging.ERROR)

# local imports
from ntap.bagofwords import TFIDF, LDA
from ntap.dic import Dictionary, DDR
from ._build import _build_design_matrix, _build_targets
from ntap.formula import _parse_formula
from ._summary import SupervisedSummary
#from ntap.supervised import summarize
#from ntap import neural_models # import LSTM, BiLSTM, FineTune

logger = logging.getLogger(__name__)



class MethodInfo:
    method_list = {
        'least_squares': {
            'backend': 'sklearn',
            'classifier': linear_model.LogisticRegression(),
            'regressor': linear_model.LinearRegression()
        },
        'svm-lin': {
            'backend': 'sklearn',
            'classifier': svm.LinearSVC(),
            'regressor': svm.LinearSVR()
        },
        'svm': {
            'backend': 'sklearn',
            'classifier': svm.SVC(),
            'regressor': svm.SVR()
        },
        'tree-ensemble': {
            'backend': 'sklearn',
            'classifier': ensemble.GradientBoostingClassifier(),
            'regressor': ensemble.GradientBoostingRegressor()
        },
        'naive-bayes': {
            'backend': 'sklearn',
            'classifier': None
        },
        'lstm': {
            'backend': 'pytorch',
            'classifier': None,
            'regressor': None
        },
        'finetune': {
            'backend': 'transformers',
            'classifier': None,
            'regressor': None}
   #'sequence': ['finetune']}
    }

    def __init__(self, method_desc, task, num_classes=None):
        self.method_desc = method_desc

        if method_desc not in self.method_list:
            #TODO: fuzzy str matching
            raise ValueError(f"{method_desc} not available. Options: "
                             f"{' '.join(list(self.method_list.keys()))}")
            return

        self.backend = self.method_list[method_desc]['backend']

        if task == 'classify':
            self.model = self.method_list[method_desc]['classifier']
            self.__check_classifier()
            if num_classes is None:
                raise ValueError("`num_classes` cannot be None when task "
                                 "is `classify`")
            else:
                self.num_classes = num_classes
        elif task == 'regress':
            self.model = self.method_list[method_desc]['regressor']
            self.__check_regressor()
        else:
            raise ValueError(f"Could not identify task parameter `{task}`")

    def __check_classifier(self):
        if self.model is None:
            raise RuntimeError(f"{self.method_desc} has no classifier implemented")

    def __check_regressor(self):
        if self.model is None:
            raise RuntimeError(f"{self.method_desc} has no regressor implemented")

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), compact=True)

#elif model_family == 'svm':
#self.obj = self.__build_objective(model_family)

class TextModel:
    """ Base class for supervised models

    TextModel instances are constructed first with a formula, and 
    optionally with a method descriptor. 

    optional parameters: - alpha - l1_ratio - tol - max_iter

    Parameters
    ----------
    formula : str

        Examples of formulae:

        * Harm ~ bilstm(text_col)
    **kwargs :
        optional arguments for specification and estimation

    """


    def __init__(self, formula, method=None):

        self.formula, self.task, self.num_classes = _parse_formula(formula)
        self.model_info = MethodInfo(method, self.task, self.num_classes)

        self.is_sklearn = (self.model_info.backend == 'sklearn')

    """
    @abc.abstractmethod
    def set_analyzer(self):
        #Set analyzer function(s) and objects necessary for each modeler class
        # idea is: method objects implement functions that return important functions
        # example: feature analysis, bias measurement, etc
        pass
    """

    def fit(self,
            data: Union[dict, pd.DataFrame],
            eval_method: str = 'cross_validate', # options: validation_set, bootstrap
            scoring_metric: str = 'f1',
            na_action: str = 'remove',
            with_optuna: bool = False,
            seed: int = 729):
        """ Fit & Evaluate Model

        Fit model to data. Default behavior will perform grid search 
        (using cross validation) to find best hyperparameters. 
        Hyperparameters to search over are defined in ntap and can be 
        accessed via the ``set_grid`` and ``get_grid`` methods (TODO). 

        """

        #validator = Validation(eval_method)

        # Validation implements methods like set_cross_val_objective
        # offers optuna functionality

        Validator = GridSearchCV
        #validator = OptunaStudy if with_optuna else GridSearchCV

        if self.is_sklearn:

            X = _build_design_matrix(self.formula, data)
            y = _build_targets(self.formula, data)

            if na_action == 'remove':
                if not sparse.issparse(X):
                    non_na_mask = ~np.isnan(X).any(axis=1)
                    X = X[non_na_mask]
                    y = y[non_na_mask]
                else:
                    non_na_mask = np.ravel(X.sum(axis=1) > 0)
                    X = X[non_na_mask]
                    y = y[non_na_mask]
            elif na_action == 'warn':
                if not sparse.issparse(X):
                    num_na = len(X[np.isnan(X).any(axis=1)])
                    if num_na > 0:
                        logger.warn("NaNs were found in feature matrix.")
                else:
                    num_na = (X.sum(axis=1) == 0).sum()
                    if num_na > 0:
                        logger.warn("Empty docs were found in sparse matrix")
            else:
                raise ValueError(f"Did not recognize na_action given: {na_action}")


            params = {'C': [0.001, 0.01, 0.1, 0.5, 0.9, 1.0]}
            validator = Validator(estimator=self.model_info.model,
                                  scoring='f1',
                                  param_grid=params).fit(X=X, y=y)
            cv_result = SupervisedSummary(validator.cv_results_,
                                          task=self.task,
                                          params=params,
                                          scoring_metric='f1', 
                                          model_info=self.model_info)
            return cv_result

        #self.set_cross_val_objective(scoring=scoring_metric, X=X, y=y)
        #study = optuna.create_study(direction='maximize', storage="sqlite:///tester.db")
        #study.optimize(self.obj, n_trials=50)
        #elif eval_method == 'validation_set':
        #raise NotImplementedError("Use cross_validate eval_method")

        return self

    def predict(self, data: pd.DataFrame):
        """ Predicts labels for a trained model

        Generate predictions from data. Data is an Iterable over strings.

        TODO
        """

        pass
        # if LHS is non-null, return score

        #y, y_hat = labels, predictions

