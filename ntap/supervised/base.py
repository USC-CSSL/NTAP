import os
import abc
import logging
import warnings
from typing import Optional, Union

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.exceptions import (ConvergenceWarning,
                                UndefinedMetricWarning)
from scipy import sparse
#import optuna
#optuna.logging.set_verbosity(optuna.logging.ERROR)

# local imports
from ntap.bagofwords import TFIDF, TopicModel
from ntap.dic import Dictionary, DDR
from ._build import FeatureGrid, _build_targets
from ._formula import _parse_formula
from ._summary import SupervisedSummary, MethodInfo
#from ntap.supervised import summarize
#from ntap import neural_models # import LSTM, BiLSTM, FineTune

logger = logging.getLogger(__name__)

#elif model_family == 'svm':
#self.obj = self.__build_objective(model_family)

class TextModel:
    """ Base class for supervised models

    Parameters
    ----------
    formula : str
        Specifies model using R model syntax. A formula for a 
        supervised model contains at least one tilde (``~``), with the 
        left-hand side the target variables (dependent variables) and the
        right-hand side the predictors. 

        NTAP defines a formula syntax for easily specifying feature, 
        embedding loading, and fine-tuning. An operation such as TFIDF
        feature extraction, LDA topic modeling, or embedding lookup is 
        performed by passing a lowercase function call, from one of the 
        following options:

        * tfidf(text_column)
        * lda(text_column)
        * ddr(text_column)

    method : str
        Specifies the method for fitting to data. Supported options are via 
        the scikit-learn package (with other PyTorch models to come!)

        * "svm"
        * "svm-lin" (linear SVM)
        * "least_squares" (linear regression/logistic regression)
        * "tree-ensemble"
    **kwargs
        Optional arguments to scikit-learn constructors, such as "C" (SVM classes)

    """


    def __init__(self, formula, method=None):

        self.formula, self.task, self.num_classes = _parse_formula(formula)
        self.method_info = MethodInfo(method, self.task, self.num_classes)
        self.feature_models = FeatureGrid(self.formula.rhs_termlist)

        self.is_sklearn = (self.method_info.backend.startswith('sklearn'))

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

        Parameters
        ----------
        data : Union[dict, pd.DataFrame]
            Object containing text data as well as any variables referenced in 
            the formula, accessible via lookup (i.e., data['my_var'])
        eval_method : {'cross_validate', 'validation_set', 'bootstrap'}
            Strategy for evaluation and hyperparameter optimization.
        scoring_metric : {'f1', 'precision', 'recall', 'accuracy', 'r2', 'mse', 'rmse'}
            Scoring metric to use during fitting and parameter selection. Note
            that metrics not used here can still be specified later when compiling
            summaries of fitted models. 
        na_action : {'remove', 'warn', 'ignore'}
            If NaNs are detected either in the rows of ``data`` or after 
            applying feature extraction, specifies the approach to take. 
        with_optuna : bool
            If True, will attempt to use optuna for hyperparameter optimization. 
            If optuna is not installed, will raise warning and use default 
            (native scikit-learn or implemented methods)
        seed : int
            Seed for controlling reproducibility
        """

        Validator = GridSearchCV
        #validator = OptunaStudy if with_optuna else GridSearchCV

        if self.is_sklearn:

            X = self.feature_models.transform(data)
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

            params = self.method_info.get_param_grid()

            scorers = {'f1': metrics.make_scorer(metrics.f1_score,
                                                 zero_division=0),
                       'precision':
                       metrics.make_scorer(metrics.precision_score,
                                           zero_division=0),
                       'recall': metrics.make_scorer(metrics.recall_score),
                       'accuracy': metrics.make_scorer(metrics.accuracy_score)}

            fitting_warnings = list()
            with warnings.catch_warnings(record=True) as w:
                validator = Validator(estimator=self.method_info.model,
                                      scoring=scorers,
                                      refit='f1',
                                      param_grid=params).fit(X=X, y=y)
                self.fitting_warnings = w

            readable_formula = " + ".join([term.name() for term in
                                           self.formula.lhs_termlist])
            if readable_formula:
                readable_formula += " ~ "
            else:
                readable_formula += "~ "

            readable_formula += " + ".join([term.name() for term in
                                           self.formula.rhs_termlist])
            summ = SupervisedSummary(readable_formula,
                                     task=self.task,
                                     method_info=self.method_info,
                                     params=params,
                                     scoring_metric=scoring_metric,
                                     warnings=self.fitting_warnings)
            if self.is_sklearn:
                summ.load_sklearn_validator(validator)
            else:
                raise NotImplementedError("Only sklearn GridSearchCV implemented")
            summ.add_feature_models(self.feature_models)
            return summ

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

