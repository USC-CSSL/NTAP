from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# CV Results
from ntap.helpers import CV_Results

import tempfile
import numpy as np
from abc import ABC, abstractmethod
import os

# disable tensorflow excessive warnings/logging
#from tensorflow.compat.v1 import logging
#logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf

class Model(ABC):
    """Model class serves as parent class for all models.

    Model class serves as abstract base class for other model classes.

    Attributes:
        optimizer: An optimizer type to be used for neural networks.
        embedding_source: A string representing embedding to be used
        in feature extraction.
    """
    def __init__(self, optimizer, embedding_source = 'glove'):
        """Init function sets optimizer and embedding source.

        Args:
            optimizer: An optimizer type (string)
            embedding_source: Embedding type (string)
        """
        super().__init__()
        self.optimizer = optimizer
        self.embedding_source = embedding_source

    @abstractmethod
    def build(self):
        """Builds a model.

        Abstract method that must be implemented by inheriting classes.
        Builds and intializes a model, depending on model requirements.
        """
        pass
    @abstractmethod
    def set_params(self):
        """Sets the parameters of class.

        Abstract method that must be implemented by inheriting classes.
        Sets the parameters of the inheriting class.
        """
        pass

    def CV(self, data, num_folds=10, num_epochs=30, comp='accuracy',
            model_dir=None, batch_size=256):
        """Runs cross validation.

        Base function that runs cross validation process.
        Can be overridden by other classes.

        Args:
            data: A Dataset object.
            num_folds: The number of cross-validation folds to be run.
            num_epochs: The number of epochs.
            comp: A comparison metric.
            model_dir: The path to where the model is saved.
            batch_size: The number of batches.
        Returns:
            A CV_Results object containing prediction score information.
        """
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
            print("Conducting Fold #", i + 1)
            model_path = os.path.join(model_dir, str(i), "cv_model")
            self.cv_model_paths[i] = model_path

            self.train(data, num_epochs=num_epochs, train_indices=train_idx.tolist(),
                    test_indices=test_idx.tolist(), model_path=model_path, batch_size=batch_size)
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
        """Evaluates predicted and true labels.

        Evaluate function that compares predicted labels to true
        labels based on the given metrics.

        Args:
            predictions: Dict of targets and predictions {target:[prediction]}.
            labels: Dict of targets and true labels {target:[true labels]}.
            num_classes: Dict of targets and num classes for each
                {target:num_classes}.
            metrics: A list of strings indicating measurement metric.

        Returns:
           List of scores for each target, based on the metrics
           specified in the metrics argument.
        """
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

    def predict(self, new_data, model_path, orig_data=None, column=None, indices=None, batch_size=256,
            retrieve=list()):
        """Predicts labels for given target(s) of dataset.

       Uses trained model to make predictions for each target on a given
       dataset.

       Args:
           new_data: A Dataset object containing test data.
           model_path: The path to saved model from training.
           orig_data: Optional, will be used to add data
               to new_data if provided.
           column: A string indicating the column name from
               orig_data to be added to new_data.
           indices: An integer indicating where to batch data. (DOUBLE CHECK)
           batch_size: An intiger indicating how many data points
               each batch should contain.

       Returns:
           A dictionary where each key is a target and the corresponding
           value is a list of predicted values.
       """
        if orig_data:
            new_data.encode_with_vocab(column, orig_data)

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
            for i, feed in enumerate(new_data.batches(self.vars,
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

    def train(self, data, num_epochs=30, batch_size=256, train_indices=None,
              test_indices=None, model_path=None):
        """Trains instance of Model.

        Trains Model object on data and saves the model to model_path.

        Args:
            data: A Dataset object containing the data to be trained on.
            num_epochs: An int indicating the number of epochs to be run.
            batch_size: An int indicating the size of each batch.
            train_indices: Optional list of ints that specify the
                indices of data to be used for training.
            test_indices: Optional list of ints that specify the
                indices of the data to be used for testing.
            model_path: A string representing the path the model will be saved.
        """
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess.run(self.init)
            _ = self.sess.run(self.vars["EmbeddingInit"],
                feed_dict={self.vars["EmbeddingPlaceholder"]: data.embedding})
            for epoch in range(num_epochs):
                epoch_loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0
                num_batches, test_batches = 0, 0
                for i, feed in enumerate(data.batches(self.vars,
                    batch_size, test=False, keep_ratio=self.rnn_dropout,
                    idx=train_indices)):
                    _, loss_val, acc = self.sess.run([self.vars["training_op"],
                        self.vars["joint_loss"], self.vars["joint_accuracy"]],
                                                     feed_dict=feed)
                    epoch_loss += loss_val
                    train_accuracy += acc
                    num_batches += 1
                for i, feed in enumerate(data.batches(self.vars,
                    batch_size, test=False, keep_ratio=self.rnn_dropout,
                    idx=test_indices)):
                    acc = self.sess.run(self.vars["joint_accuracy"], feed_dict=feed)
                    test_accuracy += acc
                    test_batches += 1

                print("Epoch {}: Loss = {:.3}, Train Accuracy = {:.3}, Test Accuracy = {:.3}"
                      .format(epoch, epoch_loss/num_batches, train_accuracy/num_batches,
                              test_accuracy/test_batches))
            if model_path is not None:
                saver.save(self.sess, model_path)
        return