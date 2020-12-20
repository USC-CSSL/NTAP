import os
import re
import pytest
import pandas as pd

from ntap.parse import TextPreprocessor
from ntap.utils import load_imdb


def test_process_lower():
    in_ = 'Lorem ipsum dolor sit amet, consectetur Adipisicing elit'
    out_ = 'lorem ipsum dolor sit amet, consectetur adipisicing elit'
    tp = TextPreprocessor('lowercase')

    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]
    assert tp.transform(in_) == out_

def test_process_hashtags():
    in_ = '#lorem ipsum #dolor sit a#met, consectetur adipisicing elit'
    out_ = 'ipsum sit a#met, consectetur adipisicing elit'
    tp = TextPreprocessor('hashtags')

    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]
    assert tp.transform(in_) == out_


def test_process_mentions():
    in_ = '@lorem ipsum @dolor sit a@met,@ consectetur @@adipisicing elit'
    out_ = 'ipsum sit a@met,@ consectetur elit'
    tp = TextPreprocessor('mentions')

    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]
    assert tp.transform(in_) == out_



def test_process_links():

    in_ = 'esse cillum http://dolore eu https://fugiat nulla pic.twitter.pariatur'
    out_ = 'esse cillum eu nulla'

    tp = TextPreprocessor('links')

    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]
    assert tp.transform(in_) == out_

def test_process_punc():

    in_ = ':Lorem,, i#$p.sum!?? - 0dolor _sit amet, /\\elit)_'
    out_ = 'Lorem ipsum 0dolor sit amet elit'

    tp = TextPreprocessor('punc')

    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]
    assert tp.transform(in_) == out_


def test_process_numeric():

    in_ = 'Lorem ipsum (8) 04-25-2012'
    out_ = 'Lorem ipsum ()'
    out_2 = 'Lorem ipsum'

    tp = TextPreprocessor('digits')

    assert tp.transform(in_) == out_
    assert tp.transform([in_]) == [out_]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_]).iloc[0]

    tp = TextPreprocessor('digits+punc')

    assert tp.transform(in_) == out_2
    assert tp.transform([in_]) == [out_2]
    assert tp.transform(pd.Series([in_])).iloc[0] == pd.Series([out_2]).iloc[0]
"""
def initialize_dataset():
    data = Dataset("/var/lib/jenkins/workspace/ntap_data/data_alm.pkl")
    data.clean("text")
    data.set_params(vocab_size=5000, mallet_path = "/var/lib/jenkins/workspace/ntap_data/mallet-2.0.8/bin/mallet", glove_path = "/var/lib/jenkins/workspace/ntap_data/glove.6B/glove.6B.300d.txt")
    return data

def initialise_rnn(data, test_case_no):
    if test_case_no==1:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==2:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==3:
        model = RNN("authority ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==4:
        model = RNN("authority ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==5:
        model = RNN("authority ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==6 or test_case_no==14:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==7:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==8:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==9:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==10:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==11:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==12:
        model = RNN("authority ~ tfidf(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==13:
        model = RNN("authority ~ lda(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==15:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ tfidf(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==16:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ lda(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model


def rnn_train(model):
    model.train(data, num_epochs = 5, model_path=".")
"""

