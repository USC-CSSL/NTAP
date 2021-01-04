import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, hstack
from patsy import dmatrices, ModelDesc
from patsy import EvalEnvironment, EvalFactor
#from patsy import ModelDesc, dmatrices, dmatrix
#from patsy.state import stateful_transform

from ntap.bagofwords import TFIDF, LDA
from ntap.dic import Dictionary, DDR


def _build_design_matrix(formula, data):

    def tfidf(text, **kwargs):
        # todo: add arg to fittable transformers (refit=False) 
        # to disable .fit method for saved models
        tfidf_obj = TFIDF(**kwargs).fit(text)
        return tfidf_obj.transform(text)

    def lda(text, **kwargs):
        lda_obj = LDA(**kwargs).fit(text)
        return lda_obj.transform(text)

    def ddr(text, dic, **kwargs):
        ddr_obj = DDR(dic, **kwargs)
        return ddr_obj.transform(text)

    vectors = dict()
    matrices = dict()

    for term in formula.rhs_termlist:
        for e in term.factors:
            state = {}
            eval_env = EvalEnvironment.capture(0)
            passes = e.memorize_passes_needed(state, eval_env)
            mat = e.eval(state, data)


            if isinstance(mat, (np.ndarray, pd.Series)) and mat.shape[1] <= 1:
                if isinstance(mat, pd.Series):
                    mat = mat.values
                vectors[e.code] = np.reshape(mat, (mat.shape[0], 1))
            elif isinstance(mat, (np.ndarray, spmatrix)):
                matrices[e.code] = mat
            else:
                raise RuntimeError("Unsupported data format: {}".format(type(mat)))

    list_of_mats = list(vectors.values()) + list(matrices.values())
    if len(list_of_mats) == 1:
        X = list_of_mats[0]
    elif len(list_of_mats) > 1:
        X = hstack(list_of_mats)
    else:
        raise RuntimeError("No features found")

    return X

def _build_targets(formula, data):

    y, _ = dmatrices(ModelDesc(formula.lhs_termlist, list()), data)
    y = np.ravel(y)
    y = np.array(y)

    return y

def set_cross_val_objective(self, scoring='f1', **kwargs):
    """ Return fn with optimization task for use by optuna """

    if self.backend == 'sklearn':
        # extract X and y
        if 'X' in kwargs and 'y' in kwargs:
            X = kwargs['X']
            y = kwargs['y']
        else:
            raise RuntimeError("Attempting to set objective for sklearn estimator; "
                               "X and y must be given as arguments")
    else:
        raise NotImplementedError("Only sklearn estimators supported")

    scorer = getattr(metrics, "{}_score".format(scoring))
    def scoring_fn(estimator, X, y):
        y_pred = estimator.predict(X)
        return scorer(y, y_pred, zero_division=0)

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

        score = cross_val_score(learner, X, y, n_jobs=-1, cv=3, scoring=scoring_fn)
        return score.mean()

    self.obj = _objective
