#from sklearn.decomposition import PCA
import pandas as pd
import sys, os, json, math
import time, pickle
from random import randint
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_text
from summarystats import analyze_targets
from neural.lstm import LSTM
from utils import tokenize
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from sklearn.model_selection import train_test_split, KFold


"""
if len(sys.argv) != 2:
    print("Usage: python run_neural.py params.json")
    exit(1)
with open(sys.argv[1], 'r') as fo:
    params = json.load(fo)
"""
params = json.load(open("params/test_lstm_hate.json", "r"))
try:
    for par in params.keys():
        locals()[par] = params[par]


except KeyError:
    print("Could not load all parameters; if you're not using a parameter, set it to None")
    exit(1)



#Preprocessing the data


############### Loading non annotated data##############
print("Loading the whole dataframe")
init_clock = time.clock()

dataframe = pd.read_pickle(data_dir + '/' + saved_data) if saved else pd.read_pickle(data_dir + '/' + all_data)
print("Whole data has {} rows and {} columns".format(dataframe.shape[0], dataframe.shape[1]))

# Preprocessing the data
if not saved:
    dataframe = preprocess_text(dataframe, "text", preprocessing, data_dir, config_text, "cleaned_data")
print("Loading data took %d seconds " % (time.clock() - init_clock))

init_clock = time.clock()
docs = [tokenize(sent.lower()) for sent in dataframe["text"].values.tolist()]
print("Tokenizing data took %d seconds " % (time.clock() - init_clock))
######## Loading annotated data ############
print("Loading annotated dataframe")
annotated_df = pd.read_pickle(data_dir + '/' + saved_train) if saved else pd.read_pickle(data_dir + '/' + dataframe_name)

print("Annotated dataframe has {} rows and {} columns".format(annotated_df.shape[0], annotated_df.shape[1]))
init_clock = time.clock()
if not saved:
    annotated_df = preprocess_text(annotated_df, text_col, preprocessing, data_dir, config_text, "cleaned_train")
#annotated_df = annotated_df.sample(frac=1)
print("Loading annotated data took %d seconds " % (time.clock() - init_clock))

"""
if len(targets) == 1:
    target = targets[0]

    sample0 = annotated_df.loc[annotated_df[target] == 0]
    sample1 = annotated_df.loc[annotated_df[target] == 1]

    ratio = float(len(sample1)) / float(len(sample0) + len(sample1))
    if  ratio > 0.7 or ratio < 0.3:
        for i in range(math.floor((len(sample0) / 2 - len(sample1)) / len(sample1))):
            annotated_df = pd.concat([annotated_df, sample1])
"""
annotated_df = annotated_df.sample(frac = 1)
print("Number of data points in the sampled data is ", len(annotated_df))

if "index" not in annotated_df.columns.values:
    annotated_df["index"] = annotated_df.index.values
indices = annotated_df["index"].values

all_targets =  annotated_df.columns.values

all_labels = np.transpose(np.array([np.array(annotated_df[target]) for target in all_targets]))
target_labels = np.transpose(np.array([np.array(annotated_df[target]) for target in targets]))
#hate_labels = np.transpose(np.array(annotated_df["hate"]))

init_clock = time.clock()
anno_docs = [tokenize(sent.lower()) for sent in annotated_df[text_col].values.tolist()]
print("Tokenizing annotated data took %d seconds " % (time.clock() - init_clock))

lstm = LSTM(hidden_size, num_layers, learning_rate, batch_size, vocab_size,
        dropout_ratio, embedding_size, pretrain)
print("Learning vocabulary of size %d" % (vocab_size))
lstm.learn_vocab(anno_docs)
vocab_size = len(lstm.vocab)
print("Converting corpus of size %d to word indices based on learned vocabulary" % len(docs))
corpus_ids = lstm.tokens_to_ids(docs)

print("Converting annotated corpus of size %d to word indices based on learned vocabulary" % len(anno_docs))
anno_ids = lstm.tokens_to_ids(anno_docs, True)

max_length = max([len(line) for line in anno_ids])
print("Max number of tokens in the annotated posts is %d" % max_length)

###################################################


#vocab_size= 100

n_outputs = 3  # 0 1 2 3 4 5
#with tf.device('/GPU:0'):
if embedding_method == "GloVe":
    embeddings = lstm.load_glove("/home/aida/neural_profiles_datadir/word_embeddings/GloVe/glove.840B.300d.txt", 1, 300)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=False, name="W")

    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
    embedding_init = W.assign(embedding_placeholder)
else:
    embedding_placeholder = tf.get_variable("embedding",
                    initializer=tf.random_uniform([vocab_size, embedding_size], -1, 1),
                    dtype=tf.float32, trainable=True,)

train_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
embed = tf.nn.embedding_lookup(embedding_placeholder, train_inputs)


def last_relevant(output, length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

keep_prob = tf.placeholder(tf.float32)
#cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

#drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
network = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(num_layers)])

task_outputs = dict()
for target in targets:
    y = tf.placeholder(tf.int64, [None], name= target)
    task_outputs[target] = y

seq_length = tf.placeholder(tf.int32, [None])
"""
rnn_outputs, states = tf.nn.dynamic_rnn(network, embed,
                                    dtype=tf.float32, sequence_length=seq_length)
states_concat = tf.concat(axis = 1, values= states)
last = last_relevant(rnn_outputs, seq_length)
"""

rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(network, network, embed,
                                    dtype=tf.float32, sequence_length=seq_length)
fw_outputs, bw_outputs = rnn_outputs
fw_state, bw_state = states

fw_last = last_relevant(fw_outputs, seq_length)
bw_last = last_relevant(fw_outputs, seq_length)

last = tf.concat([fw_last, bw_last], 1)
"""
shape = output_fb.get_shape()
output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])
"""

logits = dict()
drop_out = dict()
predictions = dict()
loss = dict()
task_accuracy = dict()
xentropy = dict()
correct = dict()
target_op = dict()

for target in targets:
    logits[target] = fully_connected(last, math.floor(hidden_size / 2))
    drop_out[target] = tf.contrib.layers.dropout(logits[target], keep_prob)
    #task_logits[target] = drop_out

    predictions[target] = fully_connected(drop_out[target], n_outputs)

    xentropy[target] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= task_outputs[target], logits=predictions[target])
    loss[target] = tf.reduce_mean(xentropy[target])

    #training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #target_op[target] = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss[target])

    task_accuracy[target] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions[target], 1), task_outputs[target]), tf.float32))
    #correct[target] = tf.nn.in_top_k(predictions[target], task_outputs[target], 1)
    #task_accuracy[target] = tf.reduce_mean(tf.cast(correct[target], tf.float32))

accuracy = task_accuracy["hate"]

for target in targets:
    loss[target] *= (1 - task_accuracy[target])
joint_loss = sum(loss.values()) / len(targets)

#training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(joint_loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(joint_loss)

#saver = tf.train.Saver()
#exit(1)

feed_dict = dict()



def splitY(y_data, feed_dict):
    for i in range(len(targets)):
        feed_dict[task_outputs[targets[i]]] = y_data[:, i]
    return feed_dict

def graph(vectors, labels, targets):
    pca = PCA(n_components=2)

    vec_components = pca.fit_transform(vectors)
    df = pd.DataFrame(data=vec_components, columns=['component 1', 'component 2'])

    labeldf = pd.DataFrame(data=labels, columns=targets)

    finalDf = pd.concat([df, labeldf], axis=1)

    finalDf.to_pickle("pca_vectors.pkl")

none = list()
qes = list()
prob = list()
acc = list()

for i in range(10):
    print("test", i)
    #train_X, test_X, train_Y, test_Y, train_idx, test_idx = train_test_split(anno_ids, target_labels, indices, test_size=0.15, random_state= randint(1, 100))
    kf = KFold(n_splits=10, random_state= randint(1, 100))
    for train_index, test_index in kf.split(np.array(anno_ids)):
        train_X, test_X = np.array(anno_ids)[train_index], np.array(anno_ids)[test_index]
        train_Y, test_Y = np.array(target_labels)[train_index], np.array(target_labels)[test_index]
        train_idx, test_idx = np.array(indices)[train_index], np.array(indices)[test_index]

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()

            #for epoch in range(epoch):

            acc_train = 0
            epoch = 0
            epoch_loss = 0

            batches = lstm.get_batches(train_X, train_Y)

            test_batches = lstm.get_batches(test_X, test_Y)

            lengths = np.array([len(line) for line in test_X])
            test_X = lstm.padding(test_X)
            test_X = np.array([np.array(line) for line in test_X])
            test_Y = np.array(test_Y)
            while True:
                epoch_loss = float(0)
                count = 0
                epoch += 1
                val_batch = randint(1, len(batches))
                for (X_batch, X_len, y_batch) in batches:
                    count += 1
                    if count == val_batch:
                        X_batch_test = X_batch
                        y_batch_test = y_batch
                        X_len_test = X_len
                        continue
                    if embedding_method == "GloVe":
                        _, loss_val, predictions_ = sess.run([training_op, joint_loss, predictions["hate"]], feed_dict= splitY(y_batch, {train_inputs: X_batch, seq_length: X_len, keep_prob: dropout_ratio, embedding_placeholder: embeddings}))

                    else:
                        _, loss_val, predictions_ = sess.run([training_op, joint_loss, predictions["hate"]], feed_dict=splitY(y_batch, {train_inputs: X_batch, seq_length: X_len, keep_prob: dropout_ratio}))

                    epoch_loss += loss_val
                if embedding_method == "GloVe":
                    acc_train = accuracy.eval(feed_dict=splitY(y_batch_test, {train_inputs: X_batch_test , seq_length: X_len_test, keep_prob: 1, embedding_placeholder: embeddings}))
                else:
                    acc_train = accuracy.eval(feed_dict=splitY(y_batch_test, {train_inputs: X_batch_test, seq_length: X_len_test, keep_prob: 1}))
                acc_test = 0
                for (X_batch, X_len, y_batch) in test_batches:
                    if embedding_method == "GloVe":
                        acc_test += accuracy.eval(
                            feed_dict=splitY(y_batch, {train_inputs: X_batch, seq_length: X_len, keep_prob: 1,
                                       embedding_placeholder: embeddings}))
                    else:
                        acc_test += accuracy.eval(feed_dict=splitY(y_batch, {train_inputs: X_batch, keep_prob: 1, seq_length: X_len}))
                acc_test = acc_test / float(len(test_batches))
                epoch_loss = epoch_loss / float(len(batches))
                print(epoch, "Train accuracy:", acc_train, "Loss: ", epoch_loss , "Test accuracy: ", acc_test)

                #if acc_test > 0.78 and acc_train > 0.98 and epoch_loss < 0.05 * len(targets):
                if epoch_loss < 0.01 or epoch > 500:
                    break
            acc.append(acc_test / float(len(test_batches)))
        #    save_path = saver.save(sess, "/tmp/model.ckpt")
        ######################## Get the hidden vectors #################################
            """
            batches = lstm.get_batches(anno_ids, target_labels)
            vectors = list()
            for (X_batch, X_len, y_batch) in batches:
                if embedding_method == "GloVe":
                    vector = sess.run(last,
                                      feed_dict={train_inputs: X_batch, seq_length: X_len, keep_prob: dropout_ratio,
                                                 embedding_placeholder: embeddings})
                else:
                    vector = sess.run(last,
                                      feed_dict={train_inputs: X_batch, seq_length: X_len, keep_prob: dropout_ratio})
                vectors.extend(vector)
            vectors = np.array(vectors)
            vectors.dump("vectors.pkl")
            # hate_labels.dump("hate.pkl")
            # graph(vectors, all_labels, all_targets)
            """
        ######################## Predict the labels for all the data#####################
            results = dict()
            for idx in range((len(anno_ids) // batch_size) + 1):
                text_batch = anno_ids[idx * batch_size: min((idx + 1) * batch_size, len(corpus_ids))]
                lengths = np.array([len(line) for line in text_batch])
                text_batch = lstm.padding(text_batch)
                text_batch = np.array([np.array(line) for line in text_batch])
                if embedding_method == "GloVe":
                    output = sess.run(predictions, feed_dict={train_inputs: text_batch, seq_length: lengths, keep_prob: 1, embedding_placeholder: embeddings})
                else:
                    output = sess.run(predictions, feed_dict={train_inputs: text_batch, seq_length: lengths, keep_prob: 1})
                for target in targets:
                    results.setdefault(target, []).extend(list(np.argmax(output[target], 1)))
            results["index"] = annotated_df["index"].tolist()
            re_dataframe = pd.DataFrame(results)
            #dataframe.to_pickle(data_dir + '/' + "-".join(target for target in targets) + "-outputs-multi.pkl")
            #print("Results saved as " + data_dir + '/' + "-".join(target for target in targets) + "-outputs-multi.pkl")
            for target in targets:
                if target == "hate":
                    annotated_df[target] = pd.to_numeric(annotated_df[target])
                    print("Analyzing results of", target)
                    true = dict()
                    false = dict()
                    all = dict()
                    precision = dict()
                    f1 = dict()
                    recall = dict()
                    checked = list()
                    for i in range(n_outputs):
                        true[i] = 0
                        all[i] = 0
                        false[i] = 0

                    for idx in test_idx:
                        if idx in checked:
                            continue
                        checked.append(idx)
                        if (re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target] == annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]).any():
                            true[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] += 1
                        else:
                            false[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] +=1
                        all[annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]] += 1

                    for i in range(n_outputs):
                        precision[i] = float(true[i]) / float(true[i] + false[i]) if true[i] + false[i] != 0 else 0
                        recall[i] = float(true[i]) / float(all[i]) if all[i] != 0 else 0
                        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0
                    none.append(f1[0])
                    qes.append(f1[1])
                    prob.append(f1[2])
                    print("none", f1[0], "ques", f1[1], "prob", f1[2])

print(float(sum(none)) / float(len(none)))
print(float(sum(qes)) / float(len(qes)))
print(float(sum(prob)) / float(len(prob)))
print(float(sum(acc)) / float(len(acc)))
results = pd.DataFrame(
    {'none': none,
     'ques': qes,
     'prob': prob,
     'acc': acc
    })
results.to_pickle("results.pkl")
print("fin")