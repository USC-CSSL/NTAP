import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, hstack
from patsy import dmatrices, ModelDesc
from patsy import EvalEnvironment, EvalFactor
#from patsy import ModelDesc, dmatrices, dmatrix
#from patsy.state import stateful_transform

from ntap.bagofwords import TFIDF, TopicModel
from ntap.dic import Dictionary, DDR

class FeatureGrid:

    def __init__(self, formula_rhs, init_objects=dict()):
        """ Initializes and maintains feature objects """

        self.termlist = formula_rhs
        self.feature_models = init_objects
        self.feature_fns = {'tfidf': getattr(self, 'tfidf'),
                            'lda': getattr(self, 'lda'),
                            'ddr': getattr(self, 'ddr')}

    def transform(self, data):

        vectors = dict()
        matrices = dict()

        for term in self.termlist:
            for e in term.factors:
                state = {}
                eval_env = EvalEnvironment.capture(0)
                eval_env = eval_env.with_outer_namespace(self.feature_fns)
                passes = e.memorize_passes_needed(state, eval_env)
                mat = e.eval(state, data)

                is_var = len(mat.shape) == 1
                if is_var:
                    if isinstance(mat, pd.Series):
                        mat = mat.values
                    vectors[e.code] = np.reshape(mat, (mat.shape[0], 1))
                    #vectors[e.code] = mat
                elif isinstance(mat, (np.ndarray, spmatrix)):
                    matrices[e.code] = mat
                else:
                    raise RuntimeError("Unsupported data format: {}".format(type(mat)))

        list_of_mats = list(vectors.values()) + list(matrices.values())

        num_sparse = len([l for l in list_of_mats if isinstance(l, spmatrix)])

        if num_sparse == 0:
            if len(list_of_mats) == 1:
                return list_of_mats[0]
            else:
                return np.concatenate(list_of_mats, axis=1)
        elif len(list_of_mats) >= 1:  # at least one sparse
            return hstack(list_of_mats, format='csr')
        else:
            print(list_of_mats)
            raise RuntimeError("No features found")
            return

    def lda(self, text, **kwargs):
        if 'lda' in self.feature_models:
            model = self.feature_models['lda']
        else:
            model = LDA(**kwargs)

        if not hasattr(model, 'mdl'):
            model.fit(text)
            self.feature_models['lda'] = model

        return model.transform(text)

    def tfidf(self, text, **kwargs):
        if 'tfidf' in self.feature_models:
            model = self.feature_models['tfidf']
        else:
            model = TFIDF(**kwargs)

        if not hasattr(model, 'tfidf_model'):
            model.fit(text)
            self.feature_models['tfidf'] = model

        return model.transform(text)

    def ddr(self, text, dic, **kwargs):
        ddr_obj = DDR(dic, **kwargs)
        return ddr_obj.transform(text)

    def __len__(self):
        return len(self.feature_models)

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
