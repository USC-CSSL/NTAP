from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import itertools, collections
from abc import ABC, abstractmethod

seed = 123  #TODO: FIX


class Model(abc):
    def __init__(self):
        super().__init__()
        self.__build()

    @abstractmethod
    def __build(self):
        pass
    @abstractmethod
    def set_params(...):
        pass

    def CV(self, data, num_folds=10):  # task='classify' ?
        skf = StratifiedKFold(n_splits=num_folds, 
                              shuffle=True,
                              random_state=seed)
            
    @abstractmethod
    def batches(self, data):
        pass

    def train(self, data, params, verbose='minimal'):  # weights?
        # assumes self.formula is set
        # self.__reset_graph()
        epoch_loss = 0.
        acc_train = 0.
        with tf.session() as sess:
            for feed_dict in self.batches(data):
                # TODO: Add verbose controls
                ## TODO: Refactor: feed_dict = self.feed_dictionary(batch, weights)
                _, loss_val = sess.run([self.training_op, self.joint_loss], feed_dict=feed_dict)
                #acc_train += self.joint_accuracy.eval(feed_dict=feed_dict)
                epoch_loss += loss_val
        return   # self.model is trained

"""
class RNN

"""
class RNN(Model):
    def __init__(self, formula, data):  # other params
        # load params
        self.__parse_formula(formula)


    def __parse_formula(self, formula):
        lhs, rhs = [s.split("+") for s in formula.split('~')]
        for target in lhs:
            target = target.strip()
            if target in data.targets:
                pass
                #print("Loading", target)
            elif target in data.data.columns:
                data.encode_targets(target, encoding='one-hot')
            else:
                raise ValueError("Failed to load {}".format(target))
        for source in rhs:
            # can't have two of (seq, bag,...)
            if source starts with "seq(":
                # get sequence of int id inputs
                print("TODO")
                if source in data.data.columns:
                    print("Try to load from data.seqs dictionary")
                else:
                    print("Could not load seq({})".format(source))
            elif source starts with "bag(":
                # multi-instance learning!
                # how to aggregate? If no param set, do rebagging with default size
                print("TODO")
            elif source in data.features:
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
            elif source in data.data.columns:
                data.encode_inputs(source)
            else:
                raise ValueError("Could not parse {}".format(source))
                
    def __train_batches(self, data, batch_size=256):
        


class SVM:
    def __init__(self, formula, data, C=1.0, class_weight=None, dual=False, penalty='l2', loss='squared_hinge', tol=0.0001, max_iter=1000):

        self.C = C
        self.class_weight = class_weight
        self.dual = dual
        self.penalty = penalty
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter

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
            elif source.startswith('tfidf'):
                text_col = source.replace('tfidf(','').strip(')')
                if text_col not in data.data.columns:
                    raise ValueError("Could not parse {}".format(source))
                    continue
                data.tfidf(text_col)
            elif source.startswith('lda'):
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
        Paramset = collections.namedtuple('Paramset', 'C class_weight dual penalty loss tol max_iter')

        def __c(a):
            if isinstance(a, list) or isinstance(a, set):
                return a
            return [a]
        for p in itertools.product(__c(self.C), __c(self.class_weight), __c(self.dual), __c(self.penalty), __c(self.loss), __c(self.tol), __c(self.max_iter)):
            param_tuple = Paramset(C=p[0], class_weight=p[1], dual=p[2], penalty=p[3], loss=p[4], tol=p[5], max_iter=p[6])
            yield param_tuple

    def __get_X_y(self, data):
        inputs = list()
        self.names = list()
        for feat in data.features:
            inputs.append(data.features[feat])
            for name in data.__feature_names[feat]:
                self.names.append("{}_{}".format(feat, name))
        X = np.concatenate(inputs, axis=1)
        targets = list()
        if len(data.targets) != 1:
            raise ValueError("Multitask not enabled; encode with LabelEncoder")
            return
        y = list(data.targets.values())[0]
        return X, y

    def __get_X(self, data):
        inputs = list()
        for feat in data.features:
            inputs.append(data.features[feat])
        X = np.concatenate(inputs, axis=1)
        return X

    def CV(self, data, num_folds=10, stratified=True):
        
        X, y = self.__get_X_y(data)
        skf = StratifiedKFold(n_splits=num_folds, 
                              shuffle=True,
                              random_state=seed)
        scores = list()
        for params in self.__grid():
            cv_scores = {"params": params}
            cv_scores["accuracy"] = list()
            for train_idx, test_idx in skf.split(X, y):
                model = LinearSVC(**params._asdict())
                train_X = X[train_idx]
                train_y = y[train_idx]
                model.fit(train_X, train_y)
                test_X = X[test_idx]
                test_y = y[test_idx]
                pred_y = model.predict(test_X)
                accuracy = sum(test_y == pred_y)/len(test_y)
                cv_scores["accuracy"].append(accuracy)
                scores.append(cv_scores)

        return self.__best_model(scores)

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
        self.best_score = (metric, best_score)
        self.best_params = best_params
        return best_score, best_params, metric
    
    def format_features(self):
        if not hasattr(self, "trained"):
            raise ValueError("SVM object has no trained obj")

        # self.features is a dict of matrices (num_features) or (n_classes, num_features)
        num_features = len(self.names)

        feature_weights = #TODO

        if self.n_classes > 2:
            features = np.zeros( (self.n_classes, num_features) )
        else:
            features = np.zeros( (num_features) )
        for _, coef in self.features.items():
            if self.n_classes == 2:
                coef = coef.reshape( (num_features,) )
            features += coef
        
        if self.n_classes > 2:
            f = pd.DataFrame(features).transpose()
            f.index = feature_names
            f.columns = self.model.classes_
        else:
            f = pd.Series(features, index=feature_names)
        return f
