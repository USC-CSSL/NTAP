import os
import abc
#from typing import int, 

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from patsy import ModelDesc, dmatrices, dmatrix
from patsy import EvalEnvironment, EvalFactor
from patsy.state import stateful_transform
from scipy.sparse import spmatrix, hstack
from scipy import stats
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# local imports
from ntap.bagofwords import TFIDF, LDA
#from ntap.formula import build_design_matrices
#from ntap import neural_models # import LSTM, BiLSTM, FineTune

class TextModel(abc.ABC):
    """ Base class for supervised models
    optional parameters: - alpha - l1_ratio - tol - max_iter
    Examples of formulae:
        - Harm ~ (bilstm|text_col)

    Optional attributes:
           optimizer: An optimizer to be used during training.
           hidden_size: The default hidden layer size. 
           dropout: The default dropout rate for all dropout layers.
           learning_rate: A float indicating the learning rate for the model.
    """

    model_types = {'feature': ['least_squares', 'ridge', 'lasso', 'elasticnet',
                                      'poisson', 'svm', 'boosting_trees'],
                   'neural': ['lstm', 'bilstm', 'finetune']}

    def __init__(self, formula, model_family):

        self.model_family = model_family
        self.formula = ModelDesc.from_formula(formula)
        self.formula.rhs_termlist = [t for t in self.formula.rhs_termlist if len(t.factors) != 0]
        self.backend = 'sklearn' if model_family in self.model_types['feature'] else 'pytorch'
        #self.obj = self.set_objective(model_family=model_family)


    """
    @abc.abstractmethod
    def set_analyzer(self):
        #Set analyzer function(s) and objects necessary for each modeler class
        # idea is: method objects implement functions that return important functions
        # example: feature analysis, bias measurement, etc
        pass
    """


    def set_cross_val_objective(self, scoring='f1', **kwargs):

        if self.backend == 'sklearn':
            # extract X and y
            if 'X' in kwargs and 'y' in kwargs:
                X = kwargs['X']
                y = kwargs['y']
            else:
                raise RunTimeError("Attempting to set objective for sklearn estimator; "
                                   "X and y must be given as arguments")
        else:
            raise NotImplementedError("Only sklearn estimators supported")

        def _objective(trial):
            params = dict() #if params is None else params
            if self.model_family == 'svm':
                params['C'] = trial.suggest_float("{}_C".format(self.model_family),
                                                  0.001, 1.0)
                weighting_param = trial.suggest_float("class_weight_proportion",
                                                      0.5, 0.999)
                params['class_weight'] = {0: 1-weighting_param, 1: weighting_param}
                learner = svm.LinearSVC(**params)

            else:
                raise NotImplementedError("Only SVM implemented")
            score = cross_val_score(learner, X, y, n_jobs=-1, cv=10, scoring=scoring)
            return score.mean()

        self.obj = _objective

""" General Model Object for Text Classifiers """
class Classifier(TextModel):

    def __init__(self, formula, model_family='least_squares', **params):
        super().__init__(formula=formula, model_family=model_family)


    def build_model(self, model_family):
        if model_family == 'least_squares':
            pass
            #self.obj = linear_model.LogisticRegression()
        elif model_family == 'svm':
            self.obj = self.__build_objective(model_family)
        elif model_family == 'boosting_trees':
            pass
            #self.model_obj = ensemble.GradientBoostingClassifier()

class Regressor(TextModel):
    """ General Model Object for Text Regressors """

    def __init__(self, formula, model_family='least_squares', **kwargs):
        super().__init__(formula, model_family)

    def build_model(self, model_family):
        if model_family == 'least_squares':
            self.model_obj = linear_model.LinearRegression()
        elif model_family == 'ridge':
            self.model_obj = linear_model.RidgeRegression()
        elif model_family == 'lasso':
            self.model_obj = linear_model.Lasso()
        elif model_family == 'elasticnet':
            self.model_obj = linear_model.ElasticNet()
        elif model_family == 'poisson':
            self.model_obj = linear_model.PoissonRegressor()
        elif model_family == 'svm':
            self.model_obj = svm.SVR()
        elif model_family == 'boosting_trees':
            self.model_obj = ensemble.GradientBoostingRegressor()
        """
        elif model_family == 'mlp':
            self.model = neural_models.MLP
        elif model_family == 'lstm':
            self.model = neural_models.LSTM
        elif model_family == 'finetune':
            self.model = neural_models.FineTune
        """

class ValidatedModel:
    def __init__(self, cv_result=None, val_set_result=None, task='classify'):
        # define metrics
        pass

    def get_all_runs(self) -> pd.DataFrame:
        pass

    def print_confusion_matrix(self):
        if self.task != 'classify':
            raise RunTimeError("Confusion matrix unavailable for regression problems")


def fit(model: TextModel,
        data: pd.DataFrame,
        eval_method: str = 'cross_validate', # options: validation_set, bootstrap
        seed: int = 729) -> ValidatedModel:
    """ Fit & Evaluate Model """

    #print(model.formula)
    #tfidf = stateful_transform(lambda body, **kwargs: TFIDF(**kwargs).transform(body))
    tfidf = lambda text_col, **kwargs: TFIDF(**kwargs).transform(text_col)
    def lda(text_col, **kwargs):
        lda_obj = LDA(**kwargs).fit(text_col) 
        return lda_obj.transform(text_col)

    sparse_matrices = dict()
    column_vecs = dict()
    matrices = dict()

    for term in model.formula.rhs_termlist:
        for e in term.factors:
            state = {}
            eval_env = EvalEnvironment.capture(0)
            passes = e.memorize_passes_needed(state, eval_env)
            mat = e.eval(state, data)
            if isinstance(mat, spmatrix):
                sparse_matrices[e.code] = mat
            elif isinstance(mat, (np.ndarray, pd.Series)) and mat.shape[1] <= 1:
                # column vec
                if isinstance(mat, pd.Series):
                    mat = mat.values
                column_vecs[e.code] = np.reshape(mat, (mat.shape[0], 1))
            elif isinstance(mat, np.ndarray):
                matrices[e.code] = mat
            else:
                raise RunTimeError("Unsupported data format: {}".format(type(mat)))


    list_of_mats = list(sparse_matrices.values()) + list(column_vecs.values())+ list(matrices.values())
    if len(list_of_mats) == 1:
        X = list_of_mats[0]
    elif len(list_of_mats) > 1:
        X = hstack(list_of_mats)
    else:
        raise RunTimeError("No features found")
    y, _ = dmatrices(ModelDesc(model.formula.lhs_termlist, list()) , data)
    y = np.ravel(y)
    y = np.array(y)

    if eval_method == 'cross_validate':
        scoring_metric = 'f1'
        model.set_cross_val_objective(scoring=scoring_metric, X=X, y=y)
        study = optuna.create_study(direction='maximize', storage="sqlite:///tester.db")
        study.optimize(model.obj, n_trials=100)

        best_params = study.best_params.items()
        best_params = ['{}: {:.2f}'.format(k, v) if isinstance(v, float)
                       else '{}: {}'.format(k, v) for k, v in best_params]

        print("Run {} trials\n"
              "Best {} ({:.3f}) with params:\n"
              "{}".format(len(study.trials), scoring_metric, study.best_value, 
                          '\n'.join(best_params)))

    elif eval_method == 'fit':
        print("Not tested")
        model.fit(X, y)

def predict(model: TextModel, data: pd.DataFrame):
    """ Given trained model (with formula specified), predicts labels """

    pass
    # if LHS is non-null, return score

    #y, y_hat = labels, predictions


