import os
import numpy as np
import pandas as pd

# 3rd party imports
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

# local imports
#from ntap import neural_models # import LSTM, BiLSTM, FineTune

class TextRegressor:
    """ General Model Object for Text Regressors """

    """

    optional parameters:
        - alpha
        - l1_ratio
        - tol
        - max_iter

    Examples of formulae:
        - Harm ~ tfidf (features arg: tfidf object; or make one)
        - Harm ~ glove (embeddding arg: GloVe object; or make one)
        - Harm ~ political_party + (tfidf|text_col)
        - Harm ~ (bilstm|text_col)

    Attributes:
           optimizer: An optimizer to be used during training.
           hidden_size: The default hidden layer size. 
           cell_type: A string indicating the type of RNN cell.
           dropout: The default dropout rate for all dropout layers.
           learning_rate: A float indicating the learning rate for the model.
    """

    model_types = {'feature_models': ['least_squares', 'ridge', 'lasso', 'elasticnet',
                                      'poisson', 'svm', 'boosting_trees', 'mlp'],
                   'sequence_models': ['lstm', 'bilstm', 'finetune']}

    def __init__(self, formula, model_family='least_squares', data=None, **kwargs):
        self.formula = self.__parse_formula(formula)
        self.__build_model(model_family)
        if data is not None:
            self.fit(data)

    def __parse_formula(self, formula_str):
        _formula = dict()
        try:
            lhs, rhs = formula_str.split('~')
        except ValueError:
            raise ValueError("Bad formula ({}): No ~ found".format(formula))
        lhs, rhs = [t.strip() for t in lhs.split('+')], [t.strip() for t in rhs.split('+')]
        _formula['targets'] = lhs
        """
        # This is data-reliant; keep separate
        data.encode_targets(target, encoding='labels')  # sparse
        """
        _formula['predictors'] = [t for t in rhs if not t.startswith('(')]
        _formula['reps'] = [t for t in rhs if t.startswith('(')]

        #features = FeatureSet(formula['reps'])
        return _formula

    def __build_model(self, model_family):
        if model_family == 'least_squares':
            self.model = linear_model.LinearRegression
        elif model_family == 'ridge':
            self.model = linear_model.RidgeRegression
        elif model_family == 'lasso':
            self.model = linear_model.Lasso
        elif model_family == 'elasticnet':
            self.model = linear_model.ElasticNet
        elif model_family == 'poisson':
            self.model = linear_model.PoissonRegressor
        elif model_family == 'svm':
            self.model = svm.SVR
        elif model_family == 'boosting_trees':
            self.model = ensemble.GradientBoostingRegressor
        """
        elif model_family == 'mlp':
            self.model = neural_models.MLP
        elif model_family == 'lstm':
            self.model = neural_models.LSTM
        elif model_family == 'finetune':
            self.model = neural_models.FineTune
        """


    def fit(self, data):
        try:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.SparseDataFrame):
                Y = data.loc[:, self.formula['targets']]
            elif isinstance(data, dict):
                Y = np.array([data[k] for k in self.formula['targets']]).T
        except KeyError:
            raise ValueError("\'data\' missing target(s): ",
                             "{}".format(self.formula['targets']))
        try:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.SparseDataFrame):
                Y = data.loc[:, self.formula['targets']]
            elif isinstance(data, dict):
                Y = np.array([data[k] for k in self.formula['targets']]).T
        except KeyError:
            raise ValueError("\'data\' missing target(s): ",
                             "{}".format(self.formula['targets']))
            if len(self.formula['predictors']) > 0:
                try:
                    X = data.loc[:, self.formula['predictors']]
                except KeyError:
                    raise ValueError("\'data\' missing predictors: "
                                     "{}".format(' '.join(self.formula['predictors'])))

        if len(Y.shape) == 1:
            Y = np.squeeze(Y.values)

        # initialize model
        compiled_model = self.model()
        self.single_fit = compiled_model.fit(X.values, y=Y)




