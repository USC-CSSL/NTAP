import os
import abc

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

# local imports
from ntap.bagofwords import TFIDF
#from ntap import neural_models # import LSTM, BiLSTM, FineTune

class TextModel(abc.ABC):
    """ Abstract class for classifier and regressor
    optional parameters: - alpha - l1_ratio - tol - max_iter
    Examples of formulae:
        - Harm ~ (bilstm|text_col)

    Optional attributes:
           optimizer: An optimizer to be used during training.
           hidden_size: The default hidden layer size. 
           cell_type: A string indicating the type of RNN cell.
           dropout: The default dropout rate for all dropout layers.
           learning_rate: A float indicating the learning rate for the model.
    """

    model_types = {'feature_models': ['least_squares', 'ridge', 'lasso', 'elasticnet',
                                      'poisson', 'svm', 'boosting_trees', 'mlp'],
                   'sequence_models': ['lstm', 'bilstm', 'finetune']}

    def __init__(self, formula, model):
        self.formula = self.__parse_formula(formula_str=formula)
        self.build_model(model_family=model)

    @abc.abstractmethod
    def build_model(self, model_family):
        pass

    def __parse_formula(self, formula_str):
        _formula = dict()
        try:
            lhs, rhs = formula_str.split('~')
        except ValueError:
            raise ValueError("Bad formula ({}): No ~ found".format(formula))
        lhs, rhs = [t.strip() for t in lhs.split('+')], [t.strip() for t in rhs.split('+')]
        _formula['targets'] = lhs
        _formula['predictors'] = [t for t in rhs if not t.startswith('(')]
        _formula['reps'] = [t for t in rhs if t.startswith('(')]

        #features = FeatureSet(formula['reps'])
        return _formula


    def fit(self, data, **kwargs):
        try:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.SparseDataFrame):
                Y = data.loc[:, self.formula['targets']].values
            elif isinstance(data, dict):
                Y = np.array([data[k] for k in self.formula['targets']]).T
        except KeyError:
            raise ValueError("\'data\' missing target(s): ",
                             "{}".format(self.formula['targets']))

        try:
            for rep_str in self.formula['reps']:
                rep_str = rep_str.strip('(').strip(')')
                transform_model, source_col = rep_str.split('|')
                text = data[source_col]
                if transform_model == 'tfidf':
                    X = TFIDF(text).X.transpose()
            #if isinstance(data, pd.DataFrame) or isinstance(data, pd.SparseDataFrame):
        except KeyError:
            raise ValueError("\'data\' missing text input(s): ",
                             "{}".format(self.formula['targets']))
            if len(self.formula['predictors']) > 0:
                try:
                    X = data.loc[:, self.formula['predictors']]
                except KeyError:
                    raise ValueError("\'data\' missing predictors: "
                                     "{}".format(' '.join(self.formula['predictors'])))

        if Y.shape[1] == 1:
            Y = Y.reshape(Y.shape[0])

        self.model = self.model().fit(X, Y)
        print(self.model.predict(X).mean())


class TextClassifier(TextModel):
    """ General Model Object for Text Classifiers """
    def __init__(self, formula, model_family='least_squares', **params):
        super().__init__(formula=formula, model=model_family)

    def build_model(self, model_family):
        if model_family == 'least_squares':
            self.model = linear_model.LogisticRegression
        elif model_family == 'svm':
            self.model = svm.LinearSVC
        elif model_family == 'boosting_trees':
            self.model = ensemble.GradientBoostingClassifier

class TextRegressor(TextModel):
    """ General Model Object for Text Regressors """

    def __init__(self, formula, model_family='least_squares', **kwargs):
        super().__init__(formula, model_family)

    def build_model(self, model_family):
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

