from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import ElasticNet, LinearRegression

# CV Results

from ntap.helpers import CV_Results

import tempfile
import numpy as np
import itertools, collections
from abc import ABC, abstractmethod
import os

# disable tensorflow excessive warnings/logging
#from tensorflow.compat.v1 import logging
#logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
from tensorflow.losses import sparse_softmax_cross_entropy as cross_ent

class Model(ABC):
    def __init__(self, optimizer, embedding_source = 'glove'):
        super().__init__()
        self.optimizer = optimizer
        self.embedding_source = embedding_source

    @abstractmethod
    def build(self):
        pass
    @abstractmethod
    def set_params(self):
        pass

    def CV(self, data, num_folds=10, num_epochs=30, comp='accuracy',
            model_dir=None):
        self.cv_model_paths = dict()
        if model_dir is None:
            model_dir = os.path.join(tempfile.gettempdir(), "tf_cv_models")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        X = np.zeros(data.num_sequences)  # arbitrary for Stratified KFold
        num_classes = len(data.targets)
        if num_classes == 1:  # LabelEncoder, not one-hot
            folder = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                  random_state=self.random_state)
            y = list(data.targets.values())[0]
        else:
            folder = KFold(n_splits=num_folds, shuffle=True,
                    random_state=self.random_state)
            y = None

        results = list()
        for i, (train_idx, test_idx) in enumerate(folder.split(X, y)):

            model_path = os.path.join(model_dir, str(i), "cv_model")
            self.cv_model_paths[i] = model_path

            self.train(data, num_epochs=num_epochs, indices=train_idx.tolist(),
                    model_path=model_path)
            y = self.predict(data, indices=test_idx.tolist(),
                    model_path=model_path)
            labels = dict()
            num_classes = dict()
            for key in y:
                var_name = key.replace("prediction-", "")
                test_y, card = data.get_labels(idx=test_idx, var=var_name)
                labels[key] = test_y
                num_classes[key] = card
            stats = self.evaluate(y, labels, num_classes)  # both dict objects
            results.append(stats)
        return CV_Results(results)
        # param grid TODO

    def evaluate(self, predictions, labels, num_classes,
            metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        for key in predictions:
            if not key.startswith("prediction-"):
                continue
            if key not in labels:
                raise ValueError("Predictions and Labels have different keys")
            stat = {"Target": key.replace("prediction-", "")}
            y, y_hat = labels[key], predictions[key]
            card = num_classes[key]
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

    def predict(self, data, model_path, indices=None, batch_size=256,
            retrieve=list()):

        if model_path is None:
            raise ValueError("predict must be called with a valid model_path argument")
        fetch_vars = {v: self.vars[v] for v in self.vars if v.startswith("prediction-")}
        if len(retrieve) > 0:
            retrieve = [r for r in retrieve if r in self.list_model_vars()]
            for r in retrieve:
                fetch_vars[r] = self.vars[r]
        fetch_vars = sorted(fetch_vars.items(), key=lambda x: x[0])

        predictions = {k: list() for k,v in fetch_vars}
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            try:
                saver.restore(self.sess, model_path)
            except Exception as e:
                print("{}; could not load saved model".format(e))
            for i, feed in enumerate(data.batches(self.vars,
                batch_size, idx=indices, test=True)):
                prediction_vars = [v for k, v in fetch_vars]
                output = self.sess.run(prediction_vars, feed_dict=feed)
                for i in range(len(output)):
                    var_name = fetch_vars[i][0]
                    outputs = output[i].tolist()
                    if var_name == "rnn_alphas":
                        lens = feed[self.vars["sequence_length"]]
                        outputs = [o[:l] for o, l in zip(outputs, lens)]
                    predictions[var_name] += outputs
        return predictions

    def train(self, data, num_epochs=30, batch_size=256, indices=None,
            model_path=None):
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess.run(self.init)
            _ = self.sess.run(self.vars["EmbeddingInit"],
                feed_dict={self.vars["EmbeddingPlaceholder"]: data.embedding})
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                for i, feed in enumerate(data.batches(self.vars,
                    batch_size, test=False, keep_ratio=self.rnn_dropout,
                    idx=indices)):
                    _, loss_val = self.sess.run([self.vars["training_op"],
                        self.vars["joint_loss"]], feed_dict=feed)
                    epoch_loss += loss_val
                    num_batches += 1
                print("Loss (Epoch {}): {:.3}".format(epoch, epoch_loss/num_batches))
            if model_path is not None:
                saver.save(self.sess, model_path)
        return

class RNN(Model):
    def __init__(self, formula, data, hidden_size=128, cell='biLSTM',
            rnn_dropout=0.5, embedding_dropout=None, optimizer='adam',
            learning_rate=0.001, rnn_pooling='last',
            embedding_source='glove', random_state=None):
        Model.__init__(self, optimizer=optimizer,
                embedding_source=embedding_source)

        self.hidden_size = hidden_size
        self.bi = cell.startswith('bi')
        self.cell_type = cell[2:] if self.bi else cell
        self.rnn_dropout = rnn_dropout
        self.embedding_dropout = embedding_dropout
        self.max_seq = data.max_len  # load from data OBJ
        self.rnn_pooling = rnn_pooling
        self.random_state = random_state
        self.learning_rate = learning_rate

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
        tf.reset_default_graph()
        self.vars["sequence_length"] = tf.placeholder(tf.int32, shape=[None],
                name="SequenceLength")
        self.vars["word_inputs"] = tf.placeholder(tf.int32, shape=[None,
            self.max_seq], name="RNNInput")

        W = tf.Variable(tf.constant(0.0, shape=[len(data.vocab), data.embed_dim]), trainable=False, name="Embed")
        self.vars["Embedding"] = tf.nn.embedding_lookup(W,
                self.vars["word_inputs"])
        self.vars["EmbeddingPlaceholder"] = tf.placeholder(tf.float32,
                shape=[len(data.vocab), data.embed_dim])
        self.vars["EmbeddingInit"] = W.assign(self.vars["EmbeddingPlaceholder"])
        self.vars["states"] = self.__build_rnn(self.vars["Embedding"],
                self.hidden_size, self.cell_type, self.bi,
                self.vars["sequence_length"])

        if self.rnn_dropout is not None:
            self.vars["keep_ratio"] = tf.placeholder(tf.float32, name="KeepRatio")
            self.vars["hidden_states"] = tf.layers.dropout(self.vars["states"], rate=self.vars["keep_ratio"], name="RNNDropout")
        else:
            self.vars["hidden_states"] = self.vars["states"]

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
        self.init = tf.global_variables_initializer()

    def list_model_vars(self):
        # return list of variable names that can be retrieved during inference
        vs = [v for v in self.vars if v.startswith("loss-")]
        vs.append("hidden_states")
        if isinstance(self.rnn_pooling, int):
            vs.append("rnn_alphas")
        return vs

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
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, att_size],
            stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))

        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        self.vars["rnn_alphas"] = alphas

        return output


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
