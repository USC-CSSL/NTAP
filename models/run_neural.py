import pandas as pd
import sys, os, json, math
import time, pickle
from random import randint
import numpy as np

from preprocess import preprocess_text
from summarystats import analyze_targets
from neural.lstm import LSTM
from utils import tokenize
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from sklearn.model_selection import train_test_split


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
#targets = [col for col in annotated_df.columns.tolist() if (col.startswith("MFQ") and not col.endswith("AVG"))]
# analyze_targets(annotated_df, targets)

print("Annotated dataframe has {} rows and {} columns".format(annotated_df.shape[0], annotated_df.shape[1]))
init_clock = time.clock()
if not saved:
    annotated_df = preprocess_text(annotated_df, text_col, preprocessing, data_dir, config_text, "cleaned_train")
#annotated_df = annotated_df.sample(frac=1)
print("Loading annotated data took %d seconds " % (time.clock() - init_clock))


if len(targets) == 1:
    target = targets[0]

    sample0 = annotated_df.loc[annotated_df[target] == 0]
    sample1 = annotated_df.loc[annotated_df[target] == 1]

    ratio = float(len(sample1)) / float(len(sample0) + len(sample1))
    if  ratio > 0.7 or ratio < 0.3:
        for i in range(math.floor((len(sample0) / 2 - len(sample1)) / len(sample1))):
            annotated_df = pd.concat([annotated_df, sample1])

annotated_df = annotated_df.sample(frac = 1)
print("Number of data points in the sampled data is ", len(annotated_df))


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

n_outputs = 2  # 0 1 2 3 4 5
#with tf.device('/GPU:0'):
if embedding_method == "GloVe":
    embeddings = lstm.load_glove("/home/aida/neural_profiles_datadir/word_embeddings/GloVe/glove.6B.300d.txt", 1, 300)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=True, name="W")

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
cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
#cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
network = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])

y = tf.placeholder(tf.int32, [None])

seq_length = tf.placeholder(tf.int32, [None])
rnn_outputs, states = tf.nn.dynamic_rnn(network, embed,
                                    dtype=tf.float32, sequence_length=seq_length)

last = last_relevant(rnn_outputs, seq_length)

logits = fully_connected(last, math.floor(hidden_size / 2), activation_fn= tf.nn.sigmoid)
drop_out = tf.contrib.layers.dropout(logits, keep_prob)

predictions = fully_connected(drop_out, n_outputs, activation_fn= None)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)


xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predictions)
loss = tf.reduce_mean(xentropy)

training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(predictions, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
#exit(1)

for target in targets:
    print(target)
    labels = annotated_df[target].values
    indices = annotated_df["id"].values

    init = tf.global_variables_initializer()

    train_X, test_X, train_Y, test_Y, train_idx, test_idx = train_test_split(anno_ids, labels, indices, test_size=0.2, random_state= randint(1, 100))

    num_epochs = 200
    with tf.Session() as sess:
        init.run()

        #for epoch in range(epoch):

        acc_train = 0
        epoch = 0
        epoch_loss = 0

        batches = lstm.get_batches(train_X, train_Y)

        lengths = np.array([len(line) for line in test_X])
        test_X = lstm.padding(test_X)
        test_X = np.array([np.array(line) for line in test_X])
        test_Y = np.array(test_Y)
        print(sum(test_Y))
        print(sum(train_Y))
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
                #print(X_batch)
                #print(X_batch.shape)
                #print(y_batch.shape)
                #print(X_len)
                if embedding_method == "GloVe":
                    _, loss_val = sess.run([training_op, loss], feed_dict={train_inputs: X_batch, y: y_batch, seq_length: X_len, keep_prob: dropout_ratio, embedding_placeholder: embeddings})
                else:
                    _, loss_val = sess.run([training_op, loss], feed_dict={train_inputs: X_batch, y: y_batch, seq_length: X_len, keep_prob: dropout_ratio})
                #print(loss_val)
                epoch_loss += loss_val
            if embedding_method == "GloVe":
                acc_train = accuracy.eval(feed_dict={train_inputs: X_batch_test, y: y_batch_test, seq_length: X_len_test, keep_prob: 1, embedding_placeholder: embeddings})
            else:
                acc_train = accuracy.eval(feed_dict={train_inputs: X_batch_test, y: y_batch_test, seq_length: X_len_test, keep_prob: 1})

            if embedding_method == "GloVe":
                acc_test = accuracy.eval(
                    feed_dict={train_inputs: test_X, y: test_Y, seq_length: lengths, keep_prob: 1,
                               embedding_placeholder: embeddings})
            else:
                acc_test = accuracy.eval(feed_dict={train_inputs: test_X, y: test_Y, keep_prob: 1, seq_length: lengths})
            print(epoch, "Train accuracy:", acc_train, "Loss: ", epoch_loss / float(count), "Test accuracy: ", acc_test)

            if acc_test > 0.7 and acc_train > 0.85:
                break
        # nh 91 72
        # nm 96 86

        save_path = saver.save(sess, "/tmp/model.ckpt")

        outputs = list()
        for idx in range((len(corpus_ids) // batch_size) + 1):
            text_batch = corpus_ids[idx * batch_size: min((idx + 1) * batch_size, len(corpus_ids))]
            lengths = np.array([len(line) for line in text_batch])
            text_batch = lstm.padding(text_batch)
            text_batch = np.array([np.array(line) for line in text_batch])
            if embedding_method == "GloVe":
                output = sess.run(predictions, feed_dict= {train_inputs: text_batch, seq_length: lengths, keep_prob: 1, embedding_placeholder: embeddings})
            else:
                output = sess.run(predictions, feed_dict={train_inputs: text_batch, seq_length: lengths, keep_prob: 1})
            outputs.extend(list(np.argmax(output, 1)))
        pickle.dump(outputs, open(data_dir + '/' + target + "-outputs.pkl", "wb"))
        print(sum(outputs))
        dataframe[target] = outputs
        tn = 0
        fn = 0
        fp = 0
        tp = 0
        checked = list()
        for idx in test_idx:
            if idx in checked:
                continue
            checked.append(idx)
            if (dataframe.loc[dataframe["index"] == idx][target] == 1).bool():
                if (annotated_df.loc[annotated_df["id"] == idx][target] == 1).any():
                    tp += 1
                else:
                    fp += 1
            else:
                if (annotated_df.loc[annotated_df["id"] == idx][target] == 1).any():
                    fn += 1
                else:
                    tn += 1
        print(len(checked), "test cases")
        print("Precision: ", float(tp)/float(tp + fp))
        print("Recall: ", float(tp) / float(tp + fn))

print("fin")