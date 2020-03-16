import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ntap.helpers import CV_Results

import numpy as np
import itertools, collections


class SVM:
    def __init__(self, formula, data, C=1.0, class_weight=None, dual=False,
            penalty='l2', loss='squared_hinge', tol=0.0001, max_iter=1000,
            random_state=None):

        self.C = C
        self.class_weight = class_weight
        self.dual = dual
        self.penalty = penalty
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        self.__parse_formula(formula, data)

        #BasePredictor.__init__(self)
        #self.n_classes = n_classes
            #self.param_grid = {"class_weight": ['balanced'],
                               #"C": [1.0]}  #np.arange(0.05, 1.0, 0.05)}

    def set_params(self, **kwargs):
        if "C" in kwargs:
            self.C = kwargs["C"]
        if "class_weight" in kwargs:
            self.class_weight = kwargs["class_weight"]
        if "dual" in kwargs:
            self.dual = kwargs["dual"]
        if "penalty" in kwargs:
            self.penalty = kwargs["penalty"]
        if "loss" in kwargs:
            self.loss = kwargs["loss"]
        if "tol" in kwargs:
            self.tol = kwargs["tol"]
        if "max_iter" in kwargs:
            self.max_iter = kwargs["max_iter"]

    def __parse_formula(self, formula, data):
        lhs, rhs = [s.split("+") for s in formula.split('~')]
        for target in lhs:
            target = target.strip()
            if target in data.targets:
                print("Loading", target)
            elif target in data.data.columns:
                data.encode_targets(target, encoding='labels')
            else:
                raise ValueError("Failed to load {}".format(target))
        inputs = list()
        for source in rhs:
            source = source.strip()
            if source in data.features:
                inputs.append(source)
            elif source == 'tfidf':
                print("Fetch tfidf from features")
            elif source == 'lda':
                print("Fetch lda from features")
            elif source == 'ddr':
                print("Write DDR method")
            elif source.startswith('tfidf('):
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
            elif source.startswith('ddr('):
                text_col = source.replace('ddr(', '').strip(')')
                data.ddr(text_col, dictionary=data.dictionary)
            else:
                raise ValueError("Could not parse {}".format(source))

    def __grid(self):
        Paramset = collections.namedtuple('Paramset', 'C class_weight dual penalty loss tol max_iter')

        def __c(a):
            if isinstance(a, list) or isinstance(a, set):
                return a
            return [a]
        for p in itertools.product(__c(self.C), __c(self.class_weight), __c(self.dual), __c(self.penalty), __c(self.loss), __c(self.tol), __c(self.max_iter)):
            param_tuple = Paramset(C=p[0], class_weight=p[1], dual=p[2], penalty=p[3], loss=p[4], tol=p[5], max_iter=p[6])
            yield param_tuple

    def __get_X(self, data):
        inputs = list()
        for feat in data.features:
            inputs.append(data.features[feat])
        X = np.concatenate(inputs, axis=1)
        return X

    def CV(self, data, num_folds=10,
            stratified=True, metric="accuracy"):
        """
        evaluate between parameter sets based on 'metric' parameter
        """
        if metric not in ["accuracy", "f1", "precision", "recall", "kappa"]:
            raise ValueError("Not a valid metric for CV: {}".format(metric))

        X = self.__get_X(data)
        y, _ = data.get_labels(idx=None)
        target = list(data.targets.keys())[0]
        print("TARGET: {}".format(target))
        skf = StratifiedKFold(n_splits=num_folds,
                              shuffle=True,
                              random_state=self.random_state)
        grid_search_results = list()
        best_index = -1
        best_score = -1.0
        for params in self.__grid():
            cv_scores = {"params": params._asdict()}

            labels = list()
            predictions = list()
            for train_idx, test_idx in skf.split(X, y):
                model = LinearSVC(**params._asdict())
                train_X = X[train_idx]
                train_y, cardinality = data.get_labels(idx=train_idx)
                model.fit(train_X, train_y)
                test_X = X[test_idx]
                test_y, _ = data.get_labels(idx=test_idx)
                pred_y = model.predict(test_X)
                labels.append(test_y)
                predictions.append(pred_y)
            performance = self.evaluate(predictions=predictions,
                    labels=labels, num_classes=cardinality, target=target)
            cv_scores["stats"] = performance
            results = [score[metric] for score in performance]
            score = np.mean(results)
            if score > best_score:
                best_score = score
                best_index = len(grid_search_results)
            grid_search_results.append(cv_scores)
        return CV_Results([grid_search_results[best_index]["stats"]])

    def evaluate(self, predictions, labels, num_classes, target,
            metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        y, y_hat = labels, predictions
        card = num_classes
        for y, y_hat in zip(predictions, labels):
            stat = {"Target": target}
            for m in metrics:
                if m == 'accuracy':
                    stat[m] = accuracy_score(y, y_hat)
                avg = 'binary' if card == 2 else 'macro'
                if m == 'precision':
                    stat[m] = precision_score(y, y_hat, average=avg)
                if m == 'recall':
                    stat[m] = recall_score(y, y_hat, average=avg)
                if m == 'f1':
                    stat[m] = f1_score(y, y_hat, average=avg)
                if m == 'kappa':
                    stat[m] = cohen_kappa_score(y, y_hat)
            stats.append(stat)
        return stats

    def train(self, data, params=None):
        if params is not None:
            if hasattr(self, "best_params"):
                params = self.best_params
            self.trained = LinearSVC(**params._asdict())
        else:
            self.trained = LinearSVC()

        X, y = self.__get_X_y(data)
        self.trained.fit(X, y)

    def predict(self, data):
        if not hasattr(self, "trained"):
            raise ValueError("Call SVM.train to train model")
            return
        X = self.__get_X(data)
        y = self.trained.predict(X)
        return y

    def __best_model(self, scores, metric='accuracy'):
        best_params = None
        best_score = 0.0
        for score in scores:  # One Grid Search
            mean = np.mean(score[metric])
            if mean > best_score:
                best_score = mean
                best_params = score["params"]
        return best_score, best_params, metric