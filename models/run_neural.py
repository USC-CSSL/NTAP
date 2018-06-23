
import pandas as pd
import sys, os, json
import time

from preprocess import preprocess_text
from summarystats import analyze_targets
from neural.lstm import LSTM
from utils import tokenize
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


"""
if __name__ == '__main__':
    init_clock = time.clock()
    if len(sys.argv) != 2:
        print("Usage: python run_neural.py params.json")
        exit(1)

    with open(sys.argv[1], 'r') as fo:
        params = json.load(fo)
    try:
        for par in params.keys():
            locals()[par] = params[par]
    except KeyError:
        print("Could not load all parameters; if you're not using a parameter, set it to None")
        exit(1)
    print("Yo")
    df = pd.read_pickle(data_dir + '/' + dataframe_name)
    targets = [col for col in df.columns.tolist() if (col.startswith("MFQ") and not col.endswith("AVG"))]
    # analyze_targets(df, targets)

    print("Dataframe has {} rows and {} columns".format(df.shape[0], df.shape[1]))

    #Preprocessing the data
    df = preprocess_text(df, text_col, preprocessing, data_dir, config_text)
    # df = df.iloc[:1000, :]  # toy problem before big problem to speed debugging
    print("Loading data took %d seconds " % (time.clock() - init_clock))
    init_clock = time.clock()
    docs = [tokenize(sent.lower()) for sent in df[text_col].values.tolist()]
    print("Tokenizing data took %d seconds " % (time.clock() - init_clock))
    lstm = LSTM(hidden_size, num_layers, learning_rate, batch_size, vocab_size,
            dropout_ratio, embedding_size, pretrain)
    print("Learning vocabulary of size %d" % (vocab_size))
    lstm.learn_vocab(docs)
    vocab_size = len(lstm.vocab)
    print("Converting corpus of size %d to word indices based on learned vocabulary" % len(docs))
    corpus_ids = lstm.tokens_to_ids(docs)
    max_length = max([len(line) for line in corpus_ids])
""" 
vocab_size= 100
embed_size = 300
n_outputs = 6  # 0 1 2 3 4 5
embeddings = tf.get_variable("embedding", 
                    initializer=tf.random_uniform([vocab_size, embed_size], -1, 1),
                    dtype=tf.float32)
sess = tf.Session()
sess.run(embeddings.initializer)
print(sess.run(embeddings))
exit(1)
train_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
network = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
#network = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

y = tf.placeholder(tf.int32, [None])
seq_length = tf.placeholder(tf.int32, [None])
rnn_outputs, states = tf.nn.dynamic_rnn(network, embed, 
                                    dtype=tf.float32, sequence_length=seq_length)
logits = fully_connected(states, n_outputs, activation_fn=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

num_epochs = 5
labels = df[targets[0]].values
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        for X_batch, X_lens, y_batch in lstm.get_batches(corpus_ids, labels):
            print(X_batch.shape)
            print(y_batch.shape)
            print(X_len)
            #sess.run(training_op, feed_dict={train_inputs: X_batch, y: y_batch, seq_length: X_lens})
        #acc_train = accuracy.eval(feed_dict={train_inputs: X_batch, y: y_batch, seq_length: X_lens})
        #print(epoch, "Train accuracy:", acc_train)
