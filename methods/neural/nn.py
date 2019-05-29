import tensorflow as tf
import numpy as np
import operator
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
from nltk import tokenize as nltk_token

def execute_training_process(model, batches, test_batches, weights, all_params):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as model.sess:
        done = False
        init.run()
        epoch = 1
        while epoch<=model.epochs:
            ## Train
            acc_train, epoch_loss = model_training(model, batches, weights)
            if epoch == model.epochs:
                done = True
            ## Test
            acc_test, test_predictions, test_labels = model_testing(model, test_batches, weights, done)

            print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                  "Train Loss: ", epoch_loss / float(len(batches)),
                  "Test accuracy: ", acc_test / float(len(test_batches)))
            epoch += 1

        f1_scores, precisions, recalls = get_precision_recall_f1_scores(model, test_predictions, test_labels)
        save_path = saver.save(model.sess, get_model_dicrectory(all_params)+"/model")

    return f1_scores, precisions, recalls

def execute_prediction_process(model, batches, data_batches, weights, savedir, all_params):
    saver = tf.train.Saver()
    with tf.Session() as model.sess:
        try:
            saver.restore(model.sess, get_model_dicrectory(all_params)+"/model")
        except Exception as e:
            print(e)
            print("No saved model. Train a model before prediction")
            exit(1)

        label_predictions = {target: np.array([]) for target in model.target_cols}
        print(len(data_batches))
        for i in range(len(data_batches)):
            feed_dict = feed_dictionary(model, data_batches[i], weights)
            for j in range(len(model.target_cols)):
                label_predictions[model.target_cols[j]] = np.append(label_predictions[model.target_cols[j]],
                                                                   model.predict[model.target_cols[j]].eval(
                                                                       feed_dict=feed_dict))
            if i % 1000 == 0 and i > 0:
                print(i)
                results = pd.DataFrame.from_dict(label_predictions)
                results.to_csv(savedir + "/predictions_" + str(i * model.batch_size) + ".csv")
                label_predictions = {target: np.array([]) for target in model.target_cols}
        results = pd.DataFrame.from_dict(label_predictions)
        results.to_csv(savedir + "/predictions_" + str(i * model.batch_size) + ".csv")

def feed_dictionary(model, batch, weights):
    feed_dict = {model.train_inputs: batch["text"],
                model.sequence_length: batch["sent_lens"],
                model.keep_prob: model.keep_ratio}
    if len(batch["label"]) > 0:
        feed_dict = splitY(model, batch["label"], feed_dict)
    else:
        feed_dict[model.keep_prob] = 1

    if model.feature:
        feed_dict[model.features] = batch["feature"]

    for t in model.target_cols:
        feed_dict[model.weights[t]] = weights[t]
    if model.pretrain:
        feed_dict[model.embedding_placeholder] = model.my_embeddings
    return feed_dict

def get_model_directory(all_params):
    return all_params["path"]["dictionary_path"]+"/NTAP_model"

def get_precision_recall_f1_scores(model, test_predictions, test_labels):
    f1_scores = dict()
    precisions = dict()
    recalls = dict()
    for i in range(len(model.target_cols)):
        score = f1_score(test_predictions[model.target_cols[i]],
                         test_labels[model.target_cols[i]],
                         average = "macro" if model.n_outputs > 2 else "binary")
        pres = precision_score(test_predictions[model.target_cols[i]],
                         test_labels[model.target_cols[i]],
                         average = "macro" if model.n_outputs > 2 else "binary")
        rec = recall_score(test_predictions[model.target_cols[i]],
                         test_labels[model.target_cols[i]],
                         average = "macro" if model.n_outputs > 2 else "binary")
        print("F1", model.target_cols[i], score,
              "Precision", model.target_cols[i], pres,
              "Recall", model.target_cols[i], rec)
        f1_scores[model.target_cols[i]] = score
        precisions[model.target_cols[i]] = pres
        recalls[model.target_cols[i]] = rec
    return f1_scores, precisions, recalls

def model_testing(model, test_batches, weights, done):
    acc_test = 0
    test_predictions = {target: np.array([]) for target in model.target_cols}
    test_labels = {target: np.array([]) for target in model.target_cols}
    for batch in test_batches:
        feed_dict = feed_dictionary(model, batch, weights)
        acc_test += model.joint_accuracy.eval(feed_dict=feed_dict)
        if done:
            for i in range(len(model.target_cols)):
                test_predictions[model.target_cols[i]] = np.append(test_predictions[model.target_cols[i]],
                                                                   model.predict[model.target_cols[i]].eval(
                                                                       feed_dict=feed_dict))
                test_labels[model.target_cols[i]] = np.append(test_labels[model.target_cols[i]],
                                                              feed_dict[
                                                                  model.task_outputs[model.target_cols[i]]])
    return acc_test, test_predictions, test_labels

def model_training(model, batches, weights):
    epoch_loss = float(0)
    acc_train = 0
    for batch in batches:
        feed_dict = feed_dictionary(model, batch, weights)
        _, loss_val = model.sess.run([model.training_op, model.joint_loss], feed_dict= feed_dict)
        acc_train += model.joint_accuracy.eval(feed_dict=feed_dict)
        epoch_loss += loss_val
    return acc_train, epoch_loss

def splitY(model, y_data, feed_dict):
    for i in range(len(model.target_cols)):
        feed_dict[model.task_outputs[model.target_cols[i]]] = y_data[:, i]
    return feed_dict
