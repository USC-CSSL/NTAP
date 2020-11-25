import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, LinearRegression

import numpy as np
import itertools, collections

class LM:
    """
    Class LM: implements a linear model with a variety of regularization options, including RIDGE, LASSO, and ElasticNet
    """
    def __init__(self, formula, data, alpha=0.0,
            l1_ratio=0.5, max_iter=1000, tol=0.001,
            random_state=None):

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.normalize_inputs = False  # TODO

        self.__parse_formula(formula, data)

    def set_params(self, **kwargs):
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        if "l1_ratio" in kwargs:
            self.l1_ratio = kwargs["l1_ratio"]
        if "tol" in kwargs:
            self.tol = kwargs["tol"]
        if "max_iter" in kwargs:
            self.max_iter = kwargs["max_iter"]

    def __parse_formula(self, formula, data):
        lhs, rhs = [s.split("+") for s in formula.split('~')]
        if len(lhs) > 1:
            raise ValueError("Multiple DVs not supported")
            return
        for target in lhs:
            target = target.strip()
            if target in data.data.columns:
                data.encode_targets(target, var_type="continuous", reset=True)
            else:
                raise ValueError("Failed to load {}".format(target))
        inputs = list()
        for source in rhs:
            source = source.strip()
            if source.startswith('tfidf('):
                text_col = source.replace('tfidf(','').strip(')')
                if text_col not in data.data.columns:
                    raise ValueError("Could not parse {}".format(source))
                    continue
                data.tfidf(text_col)
            elif source.startswith('lda('):
                text_col = source.replace('lda(','').strip(')')
                if text_col not in data.data.columns:
                    raise ValueError("Could not parse {}".format(source))
                    continue
                data.lda(text_col)
            elif source in data.data.columns:
                data.encode_inputs(source)
            else:
                raise ValueError("Could not parse {}".format(source))

    def __grid(self):
        Paramset = collections.namedtuple('Paramset', 'alpha l1_ratio tol max_iter')

        def __c(a):
            if isinstance(a, list) or isinstance(a, set):
                return a
            return [a]
        for p in itertools.product(__c(self.alpha), __c(self.l1_ratio), __c(self.tol), __c(self.max_iter)):
            param_tuple = Paramset(alpha=p[0], l1_ratio=p[1], tol=p[2], max_iter=p[3])
            yield param_tuple

    def __get_X_y(self, data):
        inputs = list()
        self.names = list()
        for feat in data.features:
            inputs.append(data.features[feat])
            for name in data.feature_names[feat]:
                self.names.append("{}_{}".format(feat, name))
        X = np.concatenate(inputs, axis=1)
        return X

    def __get_X(self, data):
        inputs = list()
        for feat in data.features:
            inputs.append(data.features[feat])
        X = np.concatenate(inputs, axis=1)
        return X

    def CV(self, data, num_folds=10, metric="r2", random_state=None):

        if random_state is not None:
            self.random_state = random_state
        X = self.__get_X(data)
        y, _ = data.get_labels(idx=None)
        folds = KFold(n_splits=num_folds,
                              shuffle=True,
                              random_state=self.random_state)
        scores = list()
        """
        TODO (Anirudh): modify metrics to include accuracy, precision, recall,
            and f1 for all folds (train and test)
            - record as much info as possible and store internally
            - store in self.cv_scores
        """
        for params in self.__grid():
            params = dict()
            cv_scores = {"params": params}
            cv_scores[metric] = list()
            # TODO: add all regression metrics
            for train_idx, test_idx in folds.split(X):
                model = LinearRegression()
                #model = ElasticNet(**params._asdict())
                train_X = X[train_idx]
                train_y = y[train_idx]
                model.fit(train_X, train_y)
                test_X = X[test_idx]
                test_y = y[test_idx]
                pred_y = model.predict(test_X)
                r2 = r2_score(test_y, pred_y)
                cv_scores[metric].append(r2) # change
            scores.append(cv_scores)
        return scores[0]

    def train(self, data, params=None):
        if params is None:
            if hasattr(self, "best_params"):
                params = self.best_params
                self.trained = ElasticNet(**params._asdict())
            else:
                self.trained = ElasticNet()
        else:
            self.trained = ElasticNet(**params._asdict())

        X, y = self.__get_X_y(data)
        self.trained.fit(X, y)

    def predict(self, data):
        if not hasattr(self, "trained"):
            raise ValueError("Call SVM.train to train model")
            return
        X = self.__get_X(data)
        y = self.trained.predict(X)
        return y

    def __best_model(self, scores, metric='r2'):
        best_params = None
        best_score = -1.0
        for score in scores:  # One Grid Search
            mean = np.mean(score[metric])
            if mean > best_score:
                best_score = mean
                best_params = score["params"]
        self.best_score = (metric, best_score)
        self.best_params = best_params
        return best_score, best_params, metric