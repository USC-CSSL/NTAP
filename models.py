from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, LinearRegression

import tempfile
import numpy as np
import itertools, collections
from abc import ABC, abstractmethod
import os
from tensorflow.compat.v1 import logging
logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

"""
def deprecated(date, instructions, warn_once=True):
    def deprecated_wrapper(func):
        return func
    return deprecated_wrapper

from tensorflow.python.util import deprecation
deprecation.deprecated = deprecated
"""

import tensorflow as tf
from tensorflow.losses import sparse_softmax_cross_entropy as cross_ent

seed = 123  #TODO: FIX

class Model(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self):
        pass
    @abstractmethod
    def set_params(self):
        pass

    def CV(self, data, num_folds=10):  # task='classify' ?

        self.cv_model_paths = dict()
        model_dir = os.path.join(tempfile.gettempdir(), "tf_cv_models")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        X = np.zeros(data.num_sequences)
        num_classes = len(data.targets)
        if num_classes == 1:  # LabelEncoder, not one-hot
            folder = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                  random_state=seed)
            y = list(data.targets.values())[0]
        else:
            folder = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            y = None

        for i, (train_idx, test_idx) in enumerate(folder.split(X, y)):
            model_path = os.path.join(model_dir, str(i))
            self.cv_model_paths[i] = model_path
            train_session = self.train(data, indices=train_idx.tolist(), 
                    model_path=model_path)
            
        # param grid TODO

    def predict(self, data, indices, model_path):  # TODO: load from file with model_path
        with self.graph.as_default:
            saver = tf.train.Saver()
            with tf.Session(graph=self.graph) as sess:
                if model_path is not None:
                    saver.restore(sess, model_path)
                for i, feed_dict in enumerate(data.test_batches(self.vars,
                    self.batch_size, idx=indices)):
                    pass  # TODO: combine test_batches and train_batches into one

                

    def train(self, data, verbose='minimal', num_epochs=None, batch_size=None,
            indices=None, model_path=None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size

        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session(graph=self.graph) as sess:
                sess.run(self.init_op)  # resetting variables?
                _ = sess.run(self.vars["EmbeddingInit"],
                    feed_dict={self.vars["EmbeddingPlaceholder"]: data.embedding})
                for epoch in range(self.num_epochs):
                    epoch_loss = 0.0
                    num_batches = 0
                    for i, feed_dict in enumerate(data.train_batches(self.vars, 
                        self.batch_size, keep_ratio=self.rnn_dropout, idx=indices)):
                        _, loss_val = sess.run([self.vars["training_op"], 
                            self.vars["joint_loss"]], feed_dict=feed_dict)
                        if verbose != 'minimal':
                            print("Batch {}: Loss {:.03}".format(i, loss_val))
                        epoch_loss += loss_val
                        num_batches += 1
                    print("Loss (Epoch {}): {:.2}".format(epoch, epoch_loss/num_batches))
                if model_path is not None:
                    saver.save(sess, model_path)
                self.train_session = sess
        return

"""
class RNN

"""
class RNN(Model):
    def __init__(self, formula, data, hidden_size=128, cell='biLSTM',
            rnn_dropout=0.5, embedding_dropout=None, optimizer='adam',
            rnn_pooling='last', embedding_source='glove', learning_rate=0.001,
            num_epochs=30, batch_size=256):
        super().__init__()

        embedding_source='glove'  # only one supported
        self.hidden_size = hidden_size
        self.bi = cell.startswith('bi')
        self.cell_type = cell[2:] if self.bi else cell
        self.rnn_dropout = rnn_dropout
        self.embedding_dropout = embedding_dropout
        self.max_seq = data.max_len  # load from data OBJ
        self.rnn_pooling = rnn_pooling
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.vars = dict() # store all network variables
        self.__parse_formula(formula, data)

        self.build(data)

    def set_params(self, **kwargs):
        print("TODO")

    def __parse_formula(self, formula, data):
        lhs, rhs = [s.split("+") for s in formula.split('~')]
        for target in lhs:
            target = target.strip()
            if target in data.targets:
                print("Target already present: {}".format(target))
            elif target in data.data.columns:
                data.encode_targets(target, encoding='labels')  # sparse
            else:
                raise ValueError("Failed to load {}".format(target))
        for source in rhs:
            # can't have two of (seq, bag,...)
            source = source.strip()
            if source.startswith("seq("):
                # get sequence of int id inputs
                text_col = source.replace("seq(", "").replace(")", "")
                data.encode_docs(text_col)
                if not hasattr(data, "embedding"):
                    data.load_embedding(text_col)
                # data stored in data.inputs[text_col]
            elif source.startswith("bag("):
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
                
    def build(self, data):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.vars["sequence_length"] = tf.placeholder(tf.int32, shape=[None],
                    name="SequenceLength")
            self.vars["word_inputs"] = tf.placeholder(tf.int32, shape=[None,
                self.max_seq], name="RNNInput")

            W = tf.Variable(tf.constant(0.0, shape=[len(data.vocab), data.embed_dim]),                      trainable=False, name="Embed")
            self.vars["Embedding"] = tf.nn.embedding_lookup(W, 
                    self.vars["word_inputs"])
            self.vars["EmbeddingPlaceholder"] = tf.placeholder(tf.float32, 
                    shape=[len(data.vocab), data.embed_dim])
            self.vars["EmbeddingInit"] = W.assign(self.vars["EmbeddingPlaceholder"])
            self.vars["hidden_states"] = self.__build_rnn(self.vars["Embedding"],
                    self.hidden_size, self.cell_type, self.bi, 
                    self.vars["sequence_length"])

            if self.rnn_dropout is not None:
                self.vars["keep_ratio"] = tf.placeholder(tf.float32, name="KeepRatio")
                self.vars["hidden_states"] = tf.layers.dropout(self.vars["hidden_states"], rate=self.vars["keep_ratio"], name="RNNDropout")

            for target in data.targets:
                n_outputs = len(data.target_names[target])
                self.vars["target-{}".format(target)] = tf.placeholder(tf.int32,
                        shape=[None], name="target-{}".format(target))
                self.vars["weights-{}".format(target)] = tf.placeholder(tf.float32,
                        shape=[n_outputs], name="weights-{}".format(target))
                logits = tf.layers.dense(self.vars["hidden_states"], n_outputs)
                weight = tf.gather(self.vars["weights-{}".format(target)],
                                   self.vars["target-{}".format(target)])
                xentropy = cross_ent(labels=self.vars["target-{}".format(target)], 
                        logits=logits, weights=weight)
                self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
                self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)

            self.vars["joint_loss"] = sum([self.vars[name] for name in self.vars if name.startswith("loss")])
            if self.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'momentum':
                opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            else:
                raise ValueError("Invalid optimizer specified")
            self.vars["training_op"] = opt.minimize(loss=self.vars["joint_loss"])
            self.init_op = tf.global_variables_initializer()

    def __build_rnn(self, inputs, hidden_size, cell_type, bi, sequences, peephole=False):
        if cell_type == 'LSTM':
            if bi:
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="ForwardRNNCell",
                          state_is_tuple=False)
                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="BackwardRNNCell",
                          state_is_tuple=False)
            else:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                          use_peepholes=peephole, name="RNNCell",
                          dtype=tf.float32)
        elif cell_type == 'GRU':
            if bi:
                fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          name="ForwardRNNCell", dtype=tf.float32)
                bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          reuse=False, name="BackwardRNNCell",
                          dtype=tf.float32)
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                          name="BackwardRNNCell", dtype=tf.float32)
        if bi:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                inputs, dtype=tf.float32, sequence_length=sequences)
            hidden_states = tf.concat(outputs, 2)  # shape (B, T, 2*h)
            state = tf.concat(states, 1)  # last unit
        else:
            hidden_states, state = tf.nn.dynamic_rnn(cell, inputs,
                    dtype=tf.float32, sequence_length=sequences)

        if isinstance(self.rnn_pooling, int):
            return self.__attention(hidden_states, self.rnn_pooling)
        elif self.rnn_pooling == 'last':  # default
            return state
        elif self.rnn_pooling == 'max':
            return tf.reduce_max(hidden_states, reduction_indices=[1])
        elif self.rnn_pooling == 'mean':
            return tf.reduce_mean(hidden_states, axis=1)
            
    def __attention(self, inputs, att_size):
        self.vars["attn"] = tf.tanh(tf.layers.dense(inputs, att_size))
        self.vars["alphas"] = tf.nn.softmax(tf.layers.dense(self.vars["attn"], 1, use_bias=False))
        word_attention = tf.reduce_sum(inputs * self.vars["alphas"], 1)
        return word_attention


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
            for name in data.feature_names[feat]:
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

    def CV(self, data, num_folds=10, stratified=True, metric="accuracy"):
        
        X, y = self.__get_X_y(data)
        skf = StratifiedKFold(n_splits=num_folds, 
                              shuffle=True,
                              random_state=seed)
        scores = list()
        """
        TODO (Anirudh): modify metrics to include accuracy, precision, recall, 
            and f1 for all folds (train and test)
            - record as much info as possible and store internally
            - store in self.cv_scores
        """
        for params in self.__grid():
            cv_scores = {"params": params}
            cv_scores["accuracy"] = list()
            # TODO: add all classification metrics
            for train_idx, test_idx in skf.split(X, y):
                model = LinearSVC(**params._asdict())
                train_X = X[train_idx]
                train_y = y[train_idx]
                model.fit(train_X, train_y)
                test_X = X[test_idx]
                test_y = y[test_idx]
                pred_y = model.predict(test_X)
                accuracy = sum(test_y == pred_y)/len(test_y) # augment
                cv_scores["accuracy"].append(accuracy) # change
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
                data.encode_targets(target, var_type="continuous")
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
        y = list(data.targets.values())[0]
        np.save("lda_vecs.npy", X)
        with open("purity_save.txt", 'w') as fo:
            fo.write('\n'.join([str(val) for val in y]))
        return X, y

    def __get_X(self, data):
        inputs = list()
        for feat in data.features:
            inputs.append(data.features[feat])
        X = np.concatenate(inputs, axis=1)
        return X

    def CV(self, data, num_folds=10, metric="r2"):
        
        X, y = self.__get_X_y(data)
        folds = KFold(n_splits=num_folds, 
                              shuffle=True,
                              random_state=seed)
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
        return self.__best_model(scores)

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
            print(score)
            mean = np.mean(score[metric])
            if mean > best_score:
                best_score = mean
                best_params = score["params"]
        self.best_score = (metric, best_score)
        self.best_params = best_params
        return best_score, best_params, metric
