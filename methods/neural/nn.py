import tensorflow as tf
import numpy as np
import operator
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
from nltk import tokenize as nltk_token

def splitY(model, y_data, feed_dict):
    for i in range(len(model.target_cols)):
        feed_dict[model.task_outputs[model.target_cols[i]]] = y_data[:, i]
    return feed_dict

def build_embedding(pretrain, train_embedding, embedding_size, vocab_size):
    if pretrain:
        embeddings_variable = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                                 trainable=train_embedding, name="W")
    else:
        embeddings_variable = tf.get_variable("embedding",
                                                initializer=tf.random_uniform(
                                                    [vocab_size, embedding_size], -1, 1),
                                                dtype=tf.float32)
    return embeddings_variable

def weight_placeholder(target_cols):
    weights = dict()
    for target in target_cols:
        weights[target] = tf.placeholder(tf.float64, [None], name=target + "_w")
    return weights

def multi_GRU(cell, hidden_size, keep_ratio, num_layers):
    if cell == "LSTM":
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
    elif cell == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
    cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_ratio)
    network = tf.contrib.rnn.MultiRNNCell([cell_drop] * num_layers)
    return network

def dynamic_rnn(cell, model, hidden, keep_prob, num_layers, embed, sequence_length):
    if model[:4] == "LSTM":
        network = multi_GRU(cell, hidden, keep_prob, num_layers)
        rnn_outputs, state = tf.nn.dynamic_rnn(network, embed,
                                               dtype=tf.float32, sequence_length=sequence_length)
        if cell == "GRU":
            state = state[0]
        else:
            state = state[0].h
    else:
        f_network = multi_GRU(cell, hidden, keep_prob, num_layers)
        b_network = multi_GRU(cell, hidden, keep_prob, num_layers)
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(f_network, b_network, embed,
                                                                dtype=tf.float32,
                                                                sequence_length=sequence_length)
        fw_outputs, bw_outputs = bi_outputs
        fw_states, bw_states = bi_states
        rnn_outputs = tf.concat([fw_outputs, bw_outputs], 2)
        if cell == "GRU":
            state = tf.concat([fw_states[0], bw_states[0]], 1)
        else:
            state = tf.concat([fw_states[0].h, bw_states[0].h], 1)
    return rnn_outputs, state

def multi_outputs(target_cols):
    outputs = dict()
    for target in target_cols:
        y = tf.placeholder(tf.int64, [None], name=target)
        outputs[target] = y
    return outputs


def pred(hidden, n_outputs, weights, task_outputs):
    logits = tf.layers.dense(hidden, n_outputs)

    weight = tf.gather(weights, task_outputs)
    xentropy = tf.losses.sparse_softmax_cross_entropy(labels=task_outputs,
                                                      logits=logits,
                                                      weights=weight)
    loss = tf.reduce_mean(xentropy)
    predicted_label = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted_label, task_outputs), tf.float32))

    return loss, predicted_label, accuracy


def drop_padding(self, output, length):
    relevant = tf.gather(output, length, axis = 1)
    return relevant

def run(model, batches, test_batches, weights, all_params):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as model.sess:
        done = False
        init.run()
        epoch = 1
        f1_scores = dict()
        precisions = dict()
        recalls = dict()
        while True:
            test_predictions = {target: np.array([]) for target in model.target_cols}
            test_labels = {target: np.array([]) for target in model.target_cols}
            ## Train
            epoch_loss = float(0)
            acc_train = 0
            for batch in batches:
                feed_dict = feed_dictionary(model, batch, weights)
                _, loss_val = model.sess.run([model.training_op, model.joint_loss], feed_dict= feed_dict)
                acc_train += model.joint_accuracy.eval(feed_dict=feed_dict)
                epoch_loss += loss_val
            epoch += 1
            if epoch == model.epochs:
                done = True
            ## Test
            acc_test = 0
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

            print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                  "Loss: ", epoch_loss / float(len(batches)),
                  "Test accuracy: ", acc_test / float(len(test_batches)))
            if done:
                #test_predictions = np.transpose(np.array([test_predictions[target] for target in model.target_cols]))
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
                if not os.path.isdir(all_params["path"]["dictionary_path"]+"/best_model"):
                    os.makedirs(all_params["path"]["dictionary_path"]+"/best_model")
                    save_path = saver.save(model.sess, all_params["path"]["dictionary_path"]+"/best_model/model")
                break
    return f1_scores, precisions, recalls

def run_pred(model, batches, data_batches, weights, savedir, all_params):
    saver = tf.train.Saver()
    with tf.Session() as model.sess:
        if not os.path.isdir(all_params["path"]["dictionary_path"]+"/best_model"):
            print("No saved model. Train a model before prediction")
        else:
            saver.restore(model.sess, all_params["path"]["dictionary_path"]+"/best_model/model")
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
    #X_batch, X_len, y_batch = batch
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

def cnn(input, filter_sizes, num_filters, keep_ratio):
    pooled_outputs = list()
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, int(input.get_shape()[2]), 1, num_filters]
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID")
        relu = tf.nn.relu(tf.nn.bias_add(conv, b))

        pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
        pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    output = tf.nn.dropout(h_pool_flat, keep_ratio)
    return output

def tokenize_data(corpus, text_col, max_length, min_length):
    #sent_tokenizer = toks[self.params["tokenize"]]
    drop = list()
    for i, row in corpus.iterrows():
        tokens = nltk_token.WordPunctTokenizer().tokenize(row[text_col].lower())
        corpus.at[i, text_col] = tokens[:min(max_length, len(tokens))]
        if len(row[text_col].split()) < min_length:
            drop.append(i)
        #tokenized_corpus = [nltk_token.WordPunctTokenizer().tokenize(sent.lower()) for sent in corpus]
    #corpus = corpus.drop(drop)
    return corpus

def learn_vocab(corpus, vocab_size):
    print("Learning vocabulary of size %d" % (vocab_size))
    tokens = dict()
    for sent in corpus:
        for token in sent:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    vocab = list(words[:vocab_size]) + ["<unk>", "<pad>"]
    return vocab

def tokens_to_ids(corpus, vocab, learn_max=True):
    print("Converting corpus of size %d to word indices based on learned vocabulary" % len(corpus))
    if vocab is None:
        raise ValueError("learn_vocab before converting tokens")

    mapping = {word: idx for idx, word in enumerate(vocab)}
    unk_idx = vocab.index("<unk>")
    for i in range(len(corpus)):
        row = corpus[i]
        for j in range(len(row)):
            try:
                corpus[i][j] = mapping[corpus[i][j]]
            except:
                corpus[i][j] = unk_idx
    if learn_max:
        max_length = max([len(line) for line in corpus])
    return corpus
