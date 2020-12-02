import os
import numpy as np
from sklearn import svm 
from sklearn import linear_model


class TextClassifier(BaseEstimator, RegressorMixin):
    """ General Model Object for Text Classifiers """
    def __init__(self, formula, model_family='least_squares', features='tfidf', **params):
        try:
            self.targets = formula_dict['target']
        except ValueError:
            print("Invalid value for \"target\" key in formula")
        try:
            self.text_input = formula_dict['text_input']
        except ValueError:
            print("Invalid value for \"text_input\" key in formula")
        #if 'predictors' in formula_dict:
            #self.predictors = formula_dict['predictors']

        if model_family == 'least_squares':
            self.model = linear_model.LogisticRegression()
        elif model_family == 'svm':
            self.model = svm.SVR()
        elif model_family == 'boosting_trees':
            self.model = ensemble.GradientBoostingClassifier()

        #self.n_classes = n_classes
            #self.param_grid = {"class_weight": ['balanced'],
                               #"C": [1.0]}  #np.arange(0.05, 1.0, 0.05)}

    def fit(self, X, y):
        # X: list of list of tokens
        # y: list of outputs
        return self.model.fit(X, y)
    def predict(self, X):
        # X: list of list of tokens
        # y: list of outputs
        return self.model.predict(X)

